#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import jetson.inference
import jetson.utils

import argparse


# parse the command line
parser = argparse.ArgumentParser()

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("--network", type=str, default="pednet", help="model to use, can be:  pednet, multiped, facenet, coco-dog, coco-bottle, coco-chair, coco-airplane")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")
parser.add_argument("--profile", type=bool, default=False, help="enable performance profiling and multiple runs of the model")
parser.add_argument("--runs", type=int, default=15, help="if profiling is enabling, the number of iterations to run")

opt, argv = parser.parse_known_args()

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

# load the object detection network
net = jetson.inference.detectNet(opt.network, argv, opt.threshold)

# enable model profiling
if opt.profile is True:
	net.EnableLayerProfiler()
else:
	opt.runs = 1

# run model inference
for i in range(opt.runs):
	if opt.runs > 1:
		print("\n/////////////////////////////////////\n// RUN {:d}\n/////////////////////////////////////".format(i))
	
	# detect objects in the image (with overlay)
	detections = net.Detect(img, width, height)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
	
	# print out timing info
	net.PrintProfilerTimes()

# save the output image with the bounding box overlays
if opt.file_out is not None:
	jetson.utils.saveImageRGBA(opt.file_out, img, width, height)


