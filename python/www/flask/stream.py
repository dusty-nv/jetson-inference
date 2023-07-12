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
import sys
import threading
import traceback

from model import Model
from jetson_utils import videoSource, videoOutput


class Stream(threading.Thread):
    """
    Thread for streaming video and applying DNN inference
    """
    def __init__(self, args):
        """
        Create a stream from input/output video sources, along with DNN models.
        """
        super().__init__()
        
        self.args = args
        self.input = videoSource(args.input, argv=sys.argv)
        self.output = videoOutput(args.output, argv=sys.argv)
        self.frames = 0
        self.models = {}
        
        # these are in the order that the overlays should be composited
        model_types = {
            'background' : args.background,
            'segmentation' : args.segmentation,
            'classification': args.classification,
            'detection': args.detection,
            'pose': args.pose,
            'action': args.action
        }
        
        for key, model in model_types.items():
            if model:
                self.models[key] = Model(key, model=model, labels=args.labels, colors=args.colors, input_layer=args.input_layer, output_layer=args.output_layer)
            
        if args.action and args.classification:
            self.models['action'].fontLine = 1
        
    def process(self):
        """
        Capture one image from the stream, process it, and output it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return
            
        for model in self.models.values():
            model.Process(img)
            
        for model in self.models.values():
            img = model.Visualize(img)

        self.output.Render(img)

        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width} x {img.height})")
   
        self.frames += 1
        
    def run(self):
        """
        Run the stream processing thread's main loop.
        """
        while True:
            try:
                self.process()
            except:
                traceback.print_exc()
                
    @staticmethod
    def usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return videoSource.Usage() + videoOutput.Usage() + Model.Usage()
        