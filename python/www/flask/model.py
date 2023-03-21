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

from jetson_inference import imageNet, detectNet, segNet, poseNet, actionNet, backgroundNet
from jetson_utils import cudaFont, cudaAllocMapped, Log


class Model:
    """
    Represents DNN models for classification, detection, pose, ect.
    """
    def __init__(self, type, model, labels='', colors='', input_layer='', output_layer='', **kwargs):
        """
        Load the model, either from a built-in pre-trained model or from a user-provided model.
        
        Parameters:
        
            type (string) -- the type of the model (classification, detection, ect)
            model (string) -- either a path to the model or name of the built-in model
            labels (string) -- path to the model's labels.txt file (optional)
            input_layer (string or dict) -- the model's input layer(s)
            output_layer (string or dict) -- the model's output layers()
        """
        self.type = type
        self.model = model
        self.enabled = True
        self.results = None
        self.frames = 0
        
        if type == 'classification':
            self.net = imageNet(model=model, labels=labels, input_blob=input_layer, output_blob=output_layer)

            if 'threshold' in kwargs:
                self.net.SetThreshold(kwargs['threshold'])
                
            if 'smoothing' in kwargs:
                self.net.SetSmoothing(kwargs['smoothing'])
                
        elif type == 'detection':
            if not output_layer:
                output_layer = {'scores': '', 'bbox': ''}
            elif isinstance(output_layer, str):
                output_layer = output_layer.split(',')
                output_layer = {'scores': output_layer[0], 'bbox': output_layer[1]}
            elif not isinstance(output_layer, dict) or output_layer.keys() < {'scores', 'bbox'}:
                raise ValueError("for detection models, output_layer should be a dict with keys 'scores' and 'bbox'")
             
            print(input_layer)
            print(output_layer)
            
            self.net = detectNet(model=model, labels=labels, colors=colors,
                                 input_blob=input_layer, 
                                 output_cvg=output_layer['scores'], 
                                 output_bbox=output_layer['bbox'])
                                 
        elif type == 'segmentation':
            self.net = segNet(model=model, labels=labels, colors=colors, input_blob=input_layer, output_blob=output_layer)
            self.overlayImg = None
        elif type == 'pose':
            self.net = poseNet(model)
        elif type == 'action':
            self.net = actionNet(model)
        elif type == 'background':
            self.net = backgroundNet(model)
        else:
            raise ValueError(f"invalid model type '{type}'")
            
        if type == 'classification' or type == 'action':
            self.font = cudaFont()
            self.fontLine = 0
            
    def Process(self, img):
        """
        Process an image with the model and return the results.
        """
        if not self.enabled:
            return
            
        if self.type == 'classification' or self.type == 'action':
            self.results = self.net.Classify(img)
        elif self.type == 'detection':
            self.results = self.net.Detect(img, overlay='none')
        elif self.type == 'segmentation':
            self.results = self.net.Process(img)
        elif self.type == 'pose':
            self.results = self.net.Process(img)
        
        self.frames += 1
        return self.results

    def Visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if not self.enabled:
            return img
            
        if results is None:
            results = self.results
                
        if self.type == 'classification' or self.type == 'action':
            if results[0] >= 0:
                str = f"{results[1] * 100:05.2f}% {self.net.GetClassLabel(results[0])}"
                self.font.OverlayText(img, img.width, img.height, str, 5, 5 + (self.fontLine * 37), self.font.White, self.font.Gray40)
        elif self.type == 'detection':
            self.net.Overlay(img, results)
        elif self.type == 'segmentation':
            if not self.overlayImg or self.overlayImg.width != img.width or self.overlayImg.height != img.height:
                self.overlayImg = cudaAllocMapped(like=img)
            self.net.Overlay(self.overlayImg, filter_mode='linear')
            return self.overlayImg
        elif self.type == 'pose':
            self.net.Overlay(img, self.results, 'links,keypoints')
        elif self.type == 'background':
            self.net.Process(img)
            
        return img
        
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
        
    @staticmethod
    def Usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return imageNet.Usage() + detectNet.Usage() + segNet.Usage() + actionNet.Usage() + poseNet.Usage() + backgroundNet.Usage() 