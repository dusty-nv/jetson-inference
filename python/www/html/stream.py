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

from jetson_inference import imageNet, detectNet, segNet, actionNet, poseNet, backgroundNet
from jetson_utils import videoSource, videoOutput, cudaFont, cudaAllocMapped


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
        elif args.detection:
            self.net = detectNet(argv=sys.argv)
        elif args.segmentation:
            self.net = segNet(argv=sys.argv)
            self.overlayImg = None
        elif args.action:
            self.net = actionNet(argv=sys.argv)
            self.font = cudaFont()
        elif args.pose:
            self.net = poseNet(argv=sys.argv)
        elif args.background:
            self.net = backgroundNet(argv=sys.argv)
            
    def process(self):
        """
        Capture one image from the stream, process it, and output it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return

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
        
        elif self.args.segmentation:
            if not self.overlayImg or self.overlayImg.width != img.width or self.overlayImg.height != img.height:
                self.overlayImg = cudaAllocMapped(like=img)
                
            self.net.Process(img)
            self.net.Overlay(self.overlayImg, filter_mode='linear')
            
            img = self.overlayImg
            
        elif self.args.pose:
            poses = self.net.Process(img, overlay="links,keypoints")

            print(f"detected {len(poses)} objects in image")

            for pose in poses:
                print(pose)
                print(pose.Keypoints)
                print("Links", pose.Links)
                        
        elif self.args.background:
            self.net.Process(img)
            
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
        return videoSource.Usage() + videoOutput.Usage() + imageNet.Usage() + detectNet.Usage() + segNet.Usage() + actionNet.Usage() + poseNet.Usage() + backgroundNet.Usage() 
        