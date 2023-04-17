#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import os
import time
import shutil

import torch
import torchvision
import threading

from jetson_inference import imageNet
from jetson_utils import cudaFont, cudaAllocMapped, Log

from utils import reshape_model


class Model(threading.Thread):
    """
    Represents a classification / image tagging model with online training.
    """
    def __init__(self, args, dataset):
        """
        Initialize the model
        """
        super().__init__()
        
        self.args = args
        self.font = cudaFont()
        self.epochs = 0
        self.dataset = dataset
        self.dataloader = None
        self.best_accuracy = 0.0
        
        self.model_train = None     # PyTorch training model
        self.model_infer = None     # TensorRT inference model

        self.input_layer = 'input_0'
        self.output_layer = 'output_0'
        
        self.training_enabled = False
        self.inference_enabled = False

        self.inference_threshold = 0.001
        self.inference_smoothing = 0.0
        
        # setup model directory
        self.model_dir = os.path.join(self.args.data, 'models')
        self.best_path = os.path.join(self.model_dir, 'model_best.pth')
        self.onnx_path = os.path.join(self.model_dir, f'{args.network}.onnx')
        
        self.labels_path = os.path.join(self.model_dir, 'labels.txt')
        self.checkpoint_path = os.path.join(self.model_dir, 'checkpoint.pth')
        
        os.makedirs(self.model_dir, exist_ok=True)
        
        # start training thread
        self.start()
     
    def Classify(self, img):
        """
        Run classification inference and return the results.
        """
        if not self.inference_enabled or self.model_infer is None:
            return
            
        self.results = self.model_infer.Classify(img)

        return self.results

    def Visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if not self.inference_enabled or self.model_infer is None:
            return
            
        if results is None:
            results = self.results
                
        if results[0] >= 0:
            str = f"{results[1] * 100:05.2f}% {self.model_infer.GetClassLabel(results[0])}"
            self.font.OverlayText(img, img.width, img.height, str, 5, 5, self.font.White, self.font.Gray40)
        
        return img
        
    def run(self):
        """
        Training thread main loop
        """
        self.model_train = torchvision.models.__dict__[self.args.network](pretrained=True)
        self.model_train = self.reshape(len(self.dataset.classes))
        
        # load previous checkpoint
        if os.path.isfile(self.checkpoint_path):
            print(f"[torch]  loading checkpoint {self.checkpoint_path}")
            
            checkpoint = torch.load(self.checkpoint_path)
            
            self.model_train.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            #self.inference_enabled = True
            
        # load TensorRT model
        self.load_inference()

        # setup data transforms
        self.dataset.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.args.net_resolution, self.args.net_resolution)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # create loss function
        if self.dataset.multi_label:
            self.criterion = torch.nn.BCEWithLogitsLoss().cuda()
        else:
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        
        # training loop
        while True:
            # wait for data and for training to be enabled
            if not self.training_enabled or len(self.dataset) == 0:
                time.sleep(1.0)
                continue
                
            print(f"[torch]  training on {len(self.dataset)} images, {len(self.dataset.classes)} classes")
            print(self.dataset.classes)
            
            # reshape the model if the number of classes changed
            if self.model_train.num_classes != len(self.dataset.classes):
                self.model_train = self.reshape(len(self.dataset.classes))
                
            # create the dataloader now that we know there's data
            if self.dataloader is None:    
                self.dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.args.batch_size, shuffle=True,
                    num_workers=self.args.workers, pin_memory=True)
            
            # train the model for one epoch
            loss, accuracy = self.train_epoch()
            
            # save the model checkpoints
            is_best = accuracy > self.best_accuracy
            self.best_accuracy = max(accuracy, self.best_accuracy)
            
            self.save_checkpoint({
                'epoch': self.epochs,
                'network': self.args.network,
                'resolution': self.args.net_resolution,
                'classes': self.dataset.classes,
                'num_classes': len(self.dataset.classes),
                'multi_label': self.dataset.multi_label,
                'state_dict': self.model_train.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'accuracy': accuracy,
                'loss': loss
            }, is_best)

            self.epochs += 1

    def train_epoch(self):
        """
        Train the model for one epoch.
        """
        self.model_train.train()
        
        acc_sum = 0.0
        loss_sum = 0.0
        img_count = 0
        
        for i, (images, target) in enumerate(self.dataloader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            
            output = self.model_train(images)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
           
            loss_sum += loss.item() * images.size(0)
            acc_sum += self.compute_accuracy(output, target) * images.size(0)
            img_count += images.size(0)
            accuracy = acc_sum / img_count
            loss = loss_sum / img_count 
            
            if (i % self.args.print_freq == 0) or (i == len(self.dataloader)-1):
                print(f"[torch]  epoch {self.epochs}  [{i}/{len(self.dataloader)}]  loss={loss:.4e}  accuracy={accuracy:.2f}")
                
        return loss, accuracy
        
    def reshape(self, num_classes):
        """
        Reshape the model (during training) for a different number of classes.
        """
        self.model_train = reshape_model(self.model_train, self.args.network, num_classes)
              
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model_train.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model_train.parameters(), lr=self.args.learning_rate, momentum=0.9, weight_decay=1e-4)
            
        return self.model_train.cuda()
  
    def compute_accuracy(self, output, target, multi_label_threshold=0.5):
        """
        Computes the accuracy of predictions vs groundtruth
        """
        with torch.no_grad():
            if self.dataset.multi_label:
                output = torch.nn.functional.sigmoid(output)
                preds = ((output >= multi_label_threshold) == target.bool())   # https://medium.com/@yrodriguezmd/tackling-the-accuracy-multi-metric-9e2356f62513
                
                # https://stackoverflow.com/a/61585551
                #output[output >= multi_label_threshold] = 1
                #output[output < multi_label_threshold] = 0
                #preds = (output == target)
            else:
                output = torch.nn.functional.softmax(output, dim=-1)
                _, preds = torch.max(output, dim=-1)
                preds = (preds == target)

            return preds.float().mean().cpu().item() * 100.0
          
    def save_checkpoint(self, state, is_best):
        """
        Save a PyTorch model checkpoint, and refresh ONNX/TensorRT if it has the best accuracy so far.
        """
        torch.save(state, self.checkpoint_path)
        
        if is_best:
            shutil.copyfile(self.checkpoint_path, self.best_path)
            print(f"[torch]  saved best model to {self.best_path}")
            self.export_onnx()
            self.load_inference()
        else:
            print(f"[torch]  saved checkpoint {self.epochs} to {self.checkpoint_path}")

    def export_onnx(self):
        """
        Export the PyTorch model to ONNX.
        """
        print(f"[torch]  exporting ONNX to {self.onnx_path}")
        
        if self.dataset.multi_label:
            model = torch.nn.Sequential(self.model_train, torch.nn.Sigmoid())
        else:
            model = torch.nn.Sequential(self.model_train, torch.nn.Softmax(1))
            
        model.eval()
        
        torch.onnx.export(
            self.model_train,
            torch.ones((1, 3, self.args.net_resolution, self.args.net_resolution)).cuda(),
            self.onnx_path,
            input_names=[self.input_layer],
            output_names=[self.output_layer],
            verbose=True)
   
        with open(self.labels_path, 'w') as file:
            file.write('\n'.join(self.dataset.classes))
        
    def load_inference(self):
        """
        Load the TensorRT model from ONNX.
        """
        if not os.path.isfile(self.onnx_path):
            self.export_onnx()
            
        self.model_infer = imageNet(model=self.onnx_path, labels=self.labels_path, input_blob=self.input_layer, output_blob=self.output_layer)
                                    
        self.model_infer.SetThreshold(self.inference_threshold)
        self.model_infer.SetSmoothing(self.inference_smoothing)

    @property
    def classification_threshold(self):
        """
        Returns the confidence threshold used during classification (inference)
        """
        return self.inference_threshold
        
    @classification_threshold.setter
    def classification_threshold(self, value):
        """
        Sets the confidence threshold used during classification (inference)
        """
        if self.model_infer:
            self.model_infer.SetThreshold(value)
            
        self.inference_threshold = value
        
    @property
    def classification_smoothing(self):
        """
        Return the temporal smoothing factor used during classification (inference)
        """
        return self.inference_smoothing
        
    @classification_smoothing.setter
    def classification_smoothing(self, value):
        """
        Return the temporal smoothing factor used during classification (inference)
        """
        if self.model_infer:
            self.model_infer.SetSmoothing(value)
            
        self.inference_smoothing = value
        
    @staticmethod
    def Usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return imageNet.Usage()
        