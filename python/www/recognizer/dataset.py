#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import os
import json
import queue
import datetime
import threading
import traceback

import torch
import numpy as np
import PIL

from jetson_utils import cudaMemcpy, saveImage
from utils import alert


class Dataset(threading.Thread, torch.utils.data.Dataset):
    """
    Class for saving multi-label image tagging datasets.
    """
    def __init__(self, args):
        """
        Create dataset object.
        """
        super().__init__()

        self.args = args
        self.classes = []             # list of class names
        self.tags = {}                # dict mapping image filename => tags
        self.num_tags = 0             # total number of labels/tags
        self.active_tags = []         # list of tags to be applied to new images
        self.multi_label = False      # true if there are multiple tags (labels) per image
        self.class_distribution = []  # number of tags for each class
        
        self.queue = queue.Queue()
        self.recording = False
        self.transform = None
        self.target_transform = None
        
        # create directory structure
        self.root_dir = self.args.data
        self.image_dir = os.path.join(self.root_dir, 'images')
        
        os.makedirs(self.image_dir, exist_ok=True)
        
        # load existing annotations
        self.tags_path = os.path.join(self.root_dir, 'tags.json')
        
        if os.path.exists(self.tags_path):
            with open(self.tags_path, 'r') as file:
                self.tags = json.load(file)
                self.update_class_labels()
                self.update_class_distribution()
                print(f"dataset -- loaded tags for {len(self.tags)} images, {len(self.classes)} from {self.tags_path}")
        
        # create a default class if necessary
        if len(self.classes) == 0:
            self.classes = ['background']
            
        # start recorder thread
        self.start()
        
    def __len__(self):
        """
        Return the size of the dataset (the number of images)
        """
        return len(self.tags)    
        
    def __getitem__(self, index):
        """
        Return (image, labels) tensors for training
        """
        key = list(self.tags.keys())[index]
        tags = self.tags[key]
        
        image = PIL.Image.open(os.path.join(self.image_dir, key)).convert('RGB')
        
        if self.multi_label:
            labels = [0] * len(self.classes)
            
            for tag in tags:
                labels[self.classes.index(tag)] = 1
                
            labels = torch.FloatTensor(labels)
        else:
            labels = torch.tensor(self.classes.index(self.tags[key][0]), dtype=torch.int64)
            
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            labels = self.target_transform(labels)
            
        return image, labels
        
    def record(self):
        """
        Record the queue of incoming images.
        """
        try:
            img, timestamp = self.queue.get(timeout=1)
        except queue.Empty:
            pass
        else:
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            filepath = os.path.join(self.image_dir, filename)

            saveImage(filepath, img, quality=85)
            self.ApplyTags(filename)
            
            del img
             
    def run(self):
        """
        Run the dataset thread's main loop for recording incoming data.
        """
        while True:
            try:
                self.record()
            except:
                traceback.print_exc()
    
    def AddImage(self, img):
        """
        Adds an image to the queue to be saved to the dataset.
        """
        if not self.recording or len(self.active_tags) == 0:
            return
            
        timestamp = datetime.datetime.now()
        img_copy = cudaMemcpy(img)
        self.queue.put((img_copy, timestamp))
        
    def Upload(self, file):
        path = os.path.join(self.image_dir, file.filename)
        print(f"/dataset/upload -- saving '{file.mimetype}' to {path}")
        file.save(path)
        self.ApplyTags(file.filename)
        return path
        
    def GetActiveTags(self):
        """
        Return a comma-separated string of the currently active labels applied to images as they are recorded.
        """
        return ','.join(self.active_tags)
        
    def SetActiveTags(self, labels):
        """
        Set the list of active labels (as a comma-separated or semicolon-separated string)
        that will be applied to incoming images as they are recorded into the dataset.
        """
        if labels:
            self.active_tags = labels.replace(';', ',').split(',')
            self.active_tags = [label.strip().lower() for label in self.active_tags]
        else:
            self.active_tags = []
            
    def ApplyTags(self, filename, tags=None, flush=True):
        """
        Apply tag annotations to the image and save them to disk (by default, the active tags will be applied)
        """
        if tags is None:
            tags = self.active_tags
            
        if len(tags) == 0:
            return
            
        self.tags[filename] = self.active_tags
        
        self.update_class_labels()
        self.update_class_distribution()
        
        if flush:
            self.SaveTags()
        
        alert(f"Dataset has {len(self.tags)} images, {len(self.classes)} classes", category='dataset')
        
    def SaveTags(self, path=''):
        """
        Flush the image tags to the JSON annotations file on disk.
        """
        if not path:
            path = self.tags_path
            
        with open(path, 'w') as file:
            json.dump(self.tags, file, indent=4)
            
    def update_class_labels(self):
        """
        Sync the list of class labels from the tag annotations.
        """
        classes = []
        multi_label = False
        
        for tags in self.tags.values():
            if len(tags) > 1:
                multi_label = True
                
            for tag in tags:
                if tag not in classes:
                    classes.append(tag)
                    
        self.classes = sorted(classes)
        self.multi_label = multi_label
        
        print(f'dataset -- class labels:  {self.classes}')
     
    def update_class_distribution(self):
        """
        Update the class distribution and total tag count.
        TODO add optional 'tags' param to incremently add at runtime.
        """
        num_tags = 0
        class_distribution = [0] * len(self.classes)
        
        for tags in self.tags.values():
            for tag in tags:
                class_distribution[self.classes.index(tag)] += 1
                num_tags += 1

        self.num_tags = num_tags
        self.class_distribution = class_distribution
        
        