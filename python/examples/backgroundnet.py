#!/usr/bin/env python3
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

import sys
import argparse

from jetson_inference import backgroundNet
from jetson_utils import (videoSource, videoOutput, loadImage, Log,
                          cudaAllocMapped, cudaMemcpy, cudaResize, cudaOverlay)

# parse the command line
parser = argparse.ArgumentParser(description="Perform background subtraction/removal and replacement.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=backgroundNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="u2net", help="pre-trained model to load (see below for options)")
parser.add_argument("--replace", type=str, default="", help="image filename to use for background replacement")
parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the background removal network
net = backgroundNet(args.network, sys.argv)

# create video sources & outputs
input = videoSource(args.input_URI, argv=sys.argv)
output = videoOutput(args.output_URI, argv=sys.argv)

# image replacement routines
if args.replace:
    img_replacement = loadImage(args.replace, format='rgba8')
    img_replacement_scaled = None
    img_output = None
    
def replaceBackground(img_input):
    global img_replacement_scaled
    global img_output
    
    if not img_replacement_scaled or img_input.shape != img_replacement_scaled.shape:
        img_replacement_scaled = cudaAllocMapped(like=img_input)
        img_output = cudaAllocMapped(like=img_input)
        cudaResize(img_replacement, img_replacement_scaled, filter=args.filter_mode)
        
    cudaMemcpy(img_output, img_replacement_scaled)
    cudaOverlay(img_input, img_output, 0, 0)
    
    return img_output


# process frames until EOS or the user exits
while True:
    # capture the next image (with alpha channel)
    img_input = input.Capture(format='rgba8')

    if img_input is None: # timeout
        continue
        
    # perform background removal
    net.Process(img_input, filter=args.filter_mode)

    # perform background replacement
    if args.replace:
        img_output = replaceBackground(img_input)
    else:
        img_output = img_input
        
    # render the image
    output.Render(img_output)

    # update the title bar
    output.SetStatus("backgroundNet {:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
