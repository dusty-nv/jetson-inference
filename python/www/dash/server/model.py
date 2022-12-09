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


class Model:
    """
    Represents DNN models for classification, detection, segmentation, ect.
    These can be either built-in models or user-provided / user-trained.
    """
    def __init__(self, server, name, type, model, labels='', input_layers='', output_layers=''):
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
        self.name = name
        self.type = type
        self.server = server

        if type == 'classification':
            self.net = imageNet(model=model, labels=labels, input_blob=input_layers, output_blob=output_layers)
            self.font = cudaFont()
        elif type == 'detection':
            if not output_layers:
                output_layers = {'scores': '', 'bbox': ''}
            elif not isinstance(output_layers, dict) or output_layers.keys() < {'scores', 'bbox'}:
                raise ValueError("for detection models, output_layers should be a dict with keys 'scores' and 'bbox'")
                
            self.net = detectNet(model=model, labels=labels, input_blob=input_layers, 
                                 output_cvg=output_layers['scores'], 
                                 output_bbox=output_layers['bbox'])
        else:
            raise ValueError(f"invalid model type '{type}'")
            
    def get_config(self):
        """
        Return a dict representation of the object.
        """
        return {
            'name' : self.name,
            'type' : self.type,
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
        """
        if self.type == 'classification':
            return self.net.Classify(img)
        elif self.type == 'detection':
            return self.net.Detect(img, overlay='none')
    
    def visualize(self, img, results):
        """
        Visualize the results on an image.
        """
        if self.type == 'classification':
            str = "{:05.2f}% {:s}".format(results[1] * 100, self.get_class_name(results[0]))
            self.font.OverlayText(img, img.width, img.height, str, 5, 5, self.font.White, self.font.Gray40)
        elif self.type == 'detection':
            self.net.Overlay(img, results)