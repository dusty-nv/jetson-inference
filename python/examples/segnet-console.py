#!/usr/bin/python3
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

# parse the command line
parser = argparse.ArgumentParser(description="Segment an image using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, default=None, nargs='?', help="filename of the output image to save")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--visualize", type=str, default="overlay", choices=["overlay", "mask"], help="visualization mode for the output image, options are:\n  'overlay' or 'mask' (default: 'overlay')")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=120.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 120.0)")
parser.add_argument('--stats', action='store_true', help="compute statistics from the segmentation mask, such as class histogram")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

# allocate the output image for the overlay/mask
img_output = jetson.utils.cudaAllocMapped(width=width, height=height, format="rgba32f")

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# process the segmentation network
net.Process(img, width, height, opt.ignore_class)

# print out timing info
net.PrintProfilerTimes()

# perform the visualization
if opt.file_out is not None:
	if opt.visualize == 'overlay':
		net.Overlay(img_output, width, height, opt.filter_mode)
	elif opt.visualize == 'mask':
		net.Mask(img_output, width, height, opt.filter_mode)

	jetson.utils.cudaDeviceSynchronize()
	jetson.utils.saveImageRGBA(opt.file_out, img_output, width, height)

# compute mask statistics
if opt.stats:
	import numpy as np
	print('computing class statistics...')

	# work with the raw classification grid dimensions
	grid_width, grid_height = net.GetGridSize()	
	num_classes = net.GetNumClasses()

	# allocate a single-channel uint8 image for the class mask
	class_mask = jetson.utils.cudaAllocMapped(width=grid_width, height=grid_height, format="gray8")

	# get the class mask (each pixel contains the classID for that grid cell)
	net.Mask(class_mask, grid_width, grid_height)

	# view as numpy array (doesn't copy data)
	mask_array = jetson.utils.cudaToNumpy(class_mask)	

	# compute the number of times each class occurs in the mask
	class_histogram, _ = np.histogram(mask_array, num_classes)

	print('grid size:   {:d}x{:d}'.format(grid_width, grid_height))
	print('num classes: {:d}'.format(num_classes))

	print('-----------------------------------------')
	print(' ID  class name        count     %')
	print('-----------------------------------------')

	for n in range(num_classes):
		percentage = float(class_histogram[n]) / float(grid_width * grid_height)
		print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, net.GetClassDesc(n), class_histogram[n], percentage)) 


