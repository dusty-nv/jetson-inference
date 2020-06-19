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
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=175.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 175.0)")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

#
# segmentation buffers
#
img_mask = None
img_overlay = None
img_composite = None

overlay = "overlay" in opt.visualize
mask = "mask" in opt.visualize

def alloc_buffers( shape, format ):
	global img_mask
	global img_overlay
	global img_composite
	global img_output

	if img_overlay is not None and img_overlay.height == shape[0] and img_overlay.width == shape[1]:
		return

	if overlay:
		img_overlay = jetson.utils.cudaAllocMapped(width=shape[1], height=shape[0], format=format)

	if mask:
		mask_downsample = 2 if overlay else 1
		img_mask = jetson.utils.cudaAllocMapped(width=shape[1]/mask_downsample, height=shape[0]/mask_downsample, format=format) 

	if overlay and mask:
		img_composite = jetson.utils.cudaAllocMapped(width=img_overlay.width+img_mask.width, height=img_overlay.height, format=format) 


# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# process frames until user exits
while True:
	# capture the next image
	img_input = input.Capture()

	# allocate the buffers
	alloc_buffers(img_input.shape, img_input.format)

	# process the segmentation network
	net.Process(img_input, ignore_class=opt.ignore_class)

	# generate the overlay
	if overlay:
		net.Overlay(img_overlay, filter_mode=opt.filter_mode)

	# generate the mask
	if mask:
		net.Mask(img_mask, filter_mode=opt.filter_mode)

	# composite the images
	if overlay and mask:
		jetson.utils.cudaOverlay(img_overlay, img_composite, 0, 0)
		jetson.utils.cudaOverlay(img_mask, img_composite, img_overlay.width, 0)

	# render the output image
	output.Render(img_composite if overlay and mask else img_overlay if overlay else img_mask)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	jetson.utils.cudaDeviceSynchronize()
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break



# TODO compute mask statistics
#if opt.stats:
#	import numpy as np
#	print('computing class statistics...')

	# work with the raw classification grid dimensions
#	grid_width, grid_height = net.GetGridSize()	
#	num_classes = net.GetNumClasses()

	# allocate a single-channel uint8 image for the class mask
#	class_mask = jetson.utils.cudaAllocMapped(width=grid_width, height=grid_height, format="gray8")

	# get the class mask (each pixel contains the classID for that grid cell)
#	net.Mask(class_mask, grid_width, grid_height)

	# view as numpy array (doesn't copy data)
#	mask_array = jetson.utils.cudaToNumpy(class_mask)	

	# compute the number of times each class occurs in the mask
#	class_histogram, _ = np.histogram(mask_array, num_classes)

#	print('grid size:   {:d}x{:d}'.format(grid_width, grid_height))
#	print('num classes: {:d}'.format(num_classes))

#	print('-----------------------------------------')
#	print(' ID  class name        count     %')
#	print('-----------------------------------------')

#	for n in range(num_classes):
#		percentage = float(class_histogram[n]) / float(grid_width * grid_height)
#		print(' {:>2d}  {:<18s} {:>3d}   {:f}'.format(n, net.GetClassDesc(n), class_histogram[n], percentage)) 

