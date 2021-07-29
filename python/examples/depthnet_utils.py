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


class depthBuffers:
    def __init__(self, args):
        self.args = args
        self.depth = None
        self.composite = None
        
        self.use_input = "input" in args.visualize
        self.use_depth = "depth" in args.visualize
            
    def Alloc(self, shape, format):
        depth_size = (shape[0] * self.args.depth_size, shape[1] * self.args.depth_size)
        composite_size = [0,0]
        
        if self.depth is not None and self.depth.height == depth_size[0] and self.depth.width == depth_size[1]:
            return
            
        if self.use_depth:
            composite_size[0] = depth_size[0]
            composite_size[1] += depth_size[1]
            
        if self.use_input:
            composite_size[0] = shape[0]
            composite_size[1] += shape[1]

        self.depth = jetson.utils.cudaAllocMapped(width=depth_size[1], height=depth_size[0], format=format)
        self.composite = jetson.utils.cudaAllocMapped(width=composite_size[1], height=composite_size[0], format=format)
        