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
from jetson_utils import Log


class Model:
    """
    Represents a DNN model
    """
    def __init__(self, server, name, type, model, labels='', input_layers='', output_layers=''):
        self.name = name
        self.type = type
        self.server = server

        if type == 'classification':
            self.net = imageNet(model=model, labels=labels, input_blob=input_layers, output_blob=output_layers)

        elif type == 'detection':
            if not output_layers:
                output_layers = {'scores': '', 'bbox': ''}
            elif not isinstance(output_layers, dict) or output_layers.keys() < {'scores', 'bbox'}:
                raise ValueError("for detection models, output_layers should be a dict with keys 'scores' and 'bbox'")
                
            self.net = detectNet(model=model, labels=labels, input_blob=input_layers, 
                                 output_cvg=output_layers['scores'], 
                                 output_bbox=output_layers['bbox'])
           
    def get_config(self):
        return {
            'name' : self.name,
            'type' : self.type,
        }