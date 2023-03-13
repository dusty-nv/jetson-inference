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

from jetson_inference import imageNet, detectNet
from jetson_utils import cudaFont, Log

from collections import deque
from pprint import pprint
from time import time



class Model:
    """
    Represents DNN models for classification, detection, segmentation, ect.
    These can be either built-in models or user-provided / user-trained.
    """
    def __init__(self, server, name, type, model, labels='', input_layers='', output_layers='', **kwargs):
        """
        Load the model, either from a built-in pre-trained model or from a user-provided model.
        
        Parameters:
        
            server (Server) -- the backend server instance
            name (string) -- the name of the model
            type (string) -- the type of the model (classification, detection, ect)
            model (string) -- either a path to the model or name of the built-in model
            labels (string) -- path to the model's labels.txt file (optional)
            input_layers (string or dict) -- the model's input layer(s)
            output_layers (string or dict) -- the model's output layers()
        """
        self.server = server
        self.name = name
        self.type = type
        self.model = model
        self.labels = labels
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.results = deque(maxlen=2)
        self.stream = kwargs.get('stream')
        self.kwargs = kwargs
        
        if type == 'classification':
            self.net = imageNet(model=model, labels=labels, input_blob=input_layers, output_blob=output_layers)
            self.font = cudaFont()
            
            if 'threshold' in kwargs:
                self.net.SetThreshold(kwargs['threshold'])
                
            if 'smoothing' in kwargs:
                self.net.SetSmoothing(kwargs['smoothing'])
                
        elif type == 'detection':
            if not output_layers:
                output_layers = {'scores': '', 'bbox': ''}
            elif not isinstance(output_layers, dict) or output_layers.keys() < {'scores', 'bbox'}:
                raise ValueError("for detection models, output_layers should be a dict with keys 'scores' and 'bbox'")
                
            self.net = detectNet(model=model, labels=labels, input_blob=input_layers, 
                                 output_cvg=output_layers['scores'], 
                                 output_bbox=output_layers['bbox'])
                                 
            if 'tracking' in kwargs:
                self.net.SetTrackingEnabled(kwargs['tracking'])
                
        else:
            raise ValueError(f"invalid model type '{type}'")
       
    def clone(self, **kwargs):
        return Model(self.server, **self.get_config(), **kwargs)
        
    def get_config(self):
        """
        Return a dict representation of the object.
        """
        return {
            'name' : self.name,
            'type' : self.type,
            'model' : self.model,
            'labels' : self.labels,
            'input_layers': self.input_layers,
            'output_layers': self.output_layers,
            **self.kwargs
        }

    def get_num_classes(self):
        """
        Get the number of classes that the model supports.
        """
        return self.net.GetNumClasses()
        
    def get_class_name(self, class_id):
        """
        Return the class name or description for the given class ID.
        """
        return self.net.GetClassDesc(class_id)
    
    def process(self, img):
        """
        Process an image with the model and return the results.
        TODO refactor event creation
        """
        from server import Event
        
        if self.type == 'classification':
            results = self.net.Classify(img)
            
            if results[0] >= 0:
                if len(self.results) > 0:
                    last_results = self.results[-1]
                else:
                    last_results = (-1, -1)
                    
                if results[0] != last_results[0]:
                    self.last_event = Event(self.stream, self, results[0], self.get_class_name(results[0]), results[1])
                    #self.server.events.append(self.last_event)
                else:
                    self.last_event.update(results[1])

        elif self.type == 'detection':
            results = self.net.Detect(img, overlay='none')
    
        #print(f"{self.name} results:")
        #pprint(results)
        
        self.results.append(results)
        return results

    def visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if results is None:
            if len(self.results) > 0:
                results = self.results[-1]
            else:
                return
                
        if self.type == 'classification':
            str = f"{results[1] * 100:05.2f}% {self.get_class_name(results[0])}"
            self.font.OverlayText(img, img.width, img.height, str, 5, 5, self.font.White, self.font.Gray40)
        elif self.type == 'detection':
            self.net.Overlay(img, results)