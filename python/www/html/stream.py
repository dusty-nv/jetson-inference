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

from jetson_inference import imageNet, detectNet, actionNet, poseNet
from jetson_utils import videoSource, videoOutput, cudaFont


class Stream(threading.Thread):
    """
    Thread for streaming video and applying DNN inference
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input = videoSource(args.input, argv=sys.argv)
        self.output = videoOutput(args.output, argv=sys.argv)
        self.frames = 0

        if args.classification:
            self.net = imageNet(argv=sys.argv)
            self.font = cudaFont()
        elif args.action:
            self.net = actionNet(argv=sys.argv)
            self.font = cudaFont()
        elif args.detection:
            self.net = detectNet(argv=sys.argv)
        elif args.pose:
            self.net = poseNet(argv=sys.argv)
            
    def process(self):
        img = self.input.Capture()
        
        if self.args.classification or self.args.action:
            classID, confidence = self.net.Classify(img)
            classLabel = self.net.GetClassLabel(classID)
            confidence *= 100.0

            print(f"{confidence:05.2f}% class #{classID} ({classLabel})")

            self.font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}", x=5, y=5, 
                                  color=self.font.White, background=self.font.Gray40)               
        elif self.args.detection:
            detections = self.net.Detect(img, overlay="box,labels,conf")

            print(f"detected {len(detections)} objects")

            for detection in detections:
                print(detection)
                
        elif self.args.pose:
            poses = self.net.Process(img, overlay="links,keypoints")

            print(f"detected {len(poses)} objects in image")

            for pose in poses:
                print(pose)
                print(pose.Keypoints)
                print("Links", pose.Links)
                
        self.output.Render(img)

        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width} x {img.height})")
   
        self.frames += 1
        
    def run(self):
        while True:
            try:
                self.process()
            except:
                traceback.print_exc()
                
    @staticmethod
    def usage():
        return imageNet.Usage() + detectNet.Usage() + actionNet.Usage() + poseNet.Usage() + videoSource.Usage() + videoOutput.Usage()
        