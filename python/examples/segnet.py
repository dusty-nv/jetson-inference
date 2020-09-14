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
import sys

from segnet_utils import *

# parse the command line
parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.segNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the segmentation network
net = jetson.inference.segNet(opt.network, sys.argv)

# set the alpha blending value
net.SetOverlayAlpha(opt.alpha)

# create buffer manager
buffers = segmentationBuffers(net, opt)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)

# process frames until user exits
while True:
	# capture the next image
	img_input = input.Capture()

	# allocate buffers for this size image
	buffers.Alloc(img_input.shape, img_input.format)

	# process the segmentation network
	net.Process(img_input, ignore_class=opt.ignore_class)

	# generate the overlay
	if buffers.overlay:
		net.Overlay(buffers.overlay, filter_mode=opt.filter_mode)

	# generate the mask
	if buffers.mask:
		net.Mask(buffers.mask, filter_mode=opt.filter_mode)

	# composite the images
	if buffers.composite:
		jetson.utils.cudaOverlay(buffers.overlay, buffers.composite, 0, 0)
		jetson.utils.cudaOverlay(buffers.mask, buffers.composite, buffers.overlay.width, 0)

	# render the output image
	output.Render(buffers.output)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	jetson.utils.cudaDeviceSynchronize()
	net.PrintProfilerTimes()

    # compute segmentation class stats
	if opt.stats:
		buffers.ComputeStats()
    
	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
