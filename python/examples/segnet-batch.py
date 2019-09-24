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
import ctypes
import sys
import os

# parse the command line
parser = argparse.ArgumentParser(description="Segment a directory of images using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("input", type=str, help="path to directory of input images")
parser.add_argument("output", type=str, default=None, nargs='?', help="desired path to output directory to save the images to")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="overlay", choices=["overlay", "mask"], help="visualization mode for the output image, options are:  'overlay' or 'mask' (default: 'overlay')")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=175.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetGlobalAlpha(opt.alpha)

# list image files
images = sorted(os.listdir(opt.input))

# process images
for img_filename in images:
	# load an image (into shared CPU/GPU memory)
	img, width, height = jetson.utils.loadImageRGBA(os.path.join(opt.input, img_filename))

	# allocate the output image for the overlay/mask
	img_output = jetson.utils.cudaAllocMapped(width * height * 4 * ctypes.sizeof(ctypes.c_float))

	# process the segmentation network
	net.Process(img, width, height, opt.ignore_class)

	# perform the visualization
	if opt.output is not None:
		if not os.path.exists(opt.output):
			os.makedirs(opt.output)

		if opt.visualize == 'overlay':
			net.Overlay(img_output, width, height, opt.filter_mode)
		elif opt.visualize == 'mask':
			net.Mask(img_output, width, height, opt.filter_mode)

		jetson.utils.cudaDeviceSynchronize()
		jetson.utils.saveImageRGBA(os.path.join(opt.output, img_filename), img_output, width, height)

	# print out timing info
	net.PrintProfilerTimes()

	# free CUDA image memory
	del img
	del img_output


