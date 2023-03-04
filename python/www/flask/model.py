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

from jetson_inference import imageNet, detectNet, poseNet, actionNet
from jetson_utils import cudaFont, Log


class Model:
    """
    Represents DNN models for classification, detection, pose, ect.
    """
    def __init__(self, type, model, labels='', input_layers='', output_layers='', **kwargs):
        """
        Load the model, either from a built-in pre-trained model or from a user-provided model.
        
        Parameters:
        
            type (string) -- the type of the model (classification, detection, ect)
            model (string) -- either a path to the model or name of the built-in model
            labels (string) -- path to the model's labels.txt file (optional)
            input_layers (string or dict) -- the model's input layer(s)
            output_layers (string or dict) -- the model's output layers()
        """
        self.type = type
        self.model = model
        self.enabled = True
        
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
        else:
            raise ValueError(f"invalid model type '{type}'")
        
    def Process(self, img):
        """
        Process an image with the model and return the results.
        """
        if not self.enabled:
            return
            
        if self.type == 'classification':
            self.results = self.net.Classify(img)
        elif self.type == 'detection':
            self.results = self.net.Detect(img, overlay='none')

        return self.results

    def Visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if not self.enabled:
            return
            
        if results is None:
            results = self.results
                
        if self.type == 'classification':
            str = f"{results[1] * 100:05.2f}% {self.net.GetClassLabel(results[0])}"
            self.font.OverlayText(img, img.width, img.height, str, 5, 5, self.font.White, self.font.Gray40)
        elif self.type == 'detection':
            self.net.Overlay(img, results)

    def IsEnabled(self):
        """
        Returns true if the model is enabled for processing, false otherwise.
        """
        return self.enabled
        
    def SetEnabled(self, enabled):
        """
        Enable/disable processing of the model.
        """
        self.enabled = enabled