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
from dataset import Dataset
from utils import alert

from jetson_utils import videoSource, videoOutput, Log


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
        self.model = None
        self.dataset = Dataset(args)
        self.model = Model(args, self.dataset)
        
    def process(self):
        """
        Capture one image from the stream, process it, and output it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return
            
        self.dataset.AddImage(img)
        
        if self.model.inference_enabled:
            self.model.Classify(img)
            self.model.Visualize(img)

        self.output.Render(img)

        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width}x{img.height})")

        self.frames += 1
        
    def run(self):
        """
        Run the stream processing thread's main loop.
        """
        while True:
            try:
                self.process()
            except:
                exc = traceback.format_exc()
                alert(exc, level='error', category='exception', duration=0)
                Log.Error(exc)
                
    @staticmethod
    def usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return videoSource.Usage() + videoOutput.Usage() + Model.Usage()
        