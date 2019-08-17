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

import argparse
import logging

import jetson.inference
import jetson.utils

# create logger.
logger = logging.getLogger("imagenet-video")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s][%(name)s] %(asctime)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# parse the command line.
parser = argparse.ArgumentParser(description="Classify video file using an image recognition DNN.",
				 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=jetson.inference.imageNet.Usage())
parser.add_argument("filename", type=str, help="filename of the video to process")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load, see below for options")
parser.add_argument("--width", type=int, default=1280, help="scale video width (default is 1280)")
parser.add_argument("--height", type=int, default=720, help="scale video height (default is 720)")
parser.add_argument("--framerate", type=str, default="5/1", help="desired video framerate (default is 5/1)")
parser.add_argument("--render", action="store_true", help="render video in a window")
parser.add_argument("--profile", action="store_true", help="show network profiler times")
opt, argv = parser.parse_known_args()

# load the recognition network.
net = jetson.inference.imageNet(opt.network, argv)

# load video file.
loader = jetson.utils.VideoSource(logger, opt.width, opt.height, opt.framerate)
loader.load(opt.filename)

# create display if rendering.
display = None
if opt.render:
    display = jetson.utils.glDisplay()

def process_sample(sample):
    buffer = sample.get_buffer()
    buf_time = buffer.pts / 1000000000.0

    img, width, height = jetson.utils.cudaFromGstSample(sample)

    # classify the image
    class_idx, confidence = net.Classify(img, width, height)

    # find the object description.
    class_desc = net.GetClassDesc(class_idx)

    if display and display.IsOpen():
        # render the image.
        display.RenderOnce(img, width, height)
        # update the title bar
        display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, 1000.0 / net.GetNetworkTime()))

    # print out the result.
    logger.info("Frame at {:f} is recognized as '{:s}' (class #{:d}) with {:f}% confidence\n".format(buf_time, class_desc, class_idx, confidence * 100))

    # print network profiler
    if opt.profile:
        net.PrintProfilerTimes()

try:
    while loader.is_loading():
        sample = loader.sample_next()
        process_sample(sample)
        loader.sample_done()
        if display and not display.IsOpen():
            break
except KeyboardInterrupt:
    print("\nUser interruption. Exiting...")

loader.stop()
