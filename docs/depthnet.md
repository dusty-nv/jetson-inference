<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="backgroundnet.md">Back</a> | <a href="pytorch-transfer-learning.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Mono Depth</sup></s></p>

# Monocular Depth with DepthNet
Depth sensing is useful for tasks such as mapping, navigation and obstacle detection, however it historically required a stereo camera or RGB-D camera.  There are now DNNs that are able to infer relative depth from a single monocular image (aka mono depth).  See the [MIT FastDepth](https://arxiv.org/abs/1903.03273) paper for one such approach to accomplishing this using Fully-Convolutional Networks (FCNs).

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/depthnet-0.jpg">

The [`depthNet`](../c/depthNet.h) object accepts a single color image as input, and outputs the depth map.  The depth map is colorized for visualization, but the raw [depth field](#getting-the-raw-depth-field) is also accessible for directly accessing the depths. [`depthNet`](../c/depthNet.h) is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#depthNet) and [C++](../c/depthNet.h).

As examples of using the `depthNet` class, we provide sample programs for C++ and Python:

- [`depthnet.cpp`](../examples/depthnet/depthnet.cpp) (C++) 
- [`depthnet.py`](../python/examples/depthnet.py) (Python) 

These samples are able to infer depth from images, videos, and camera feeds.  For more info about the various types of input/output streams supported, see the [Camera Streaming and Multimedia](aux-streaming.md) page.

## Mono Depth on Images

First, let's try running the `depthnet` sample on some examples images.  In addition to the input/output paths, there are some additional command-line options that are optional:

- optional `--network` flag which changes the depth model being used (recommended default is `fcn-mobilenet`).
- optional `--visualize` flag which can be comma-separated combinations of `input`, `depth`
	- The default is `--visualize=input,depth` which displays the input and depth images side-by-side
	- To view only the depth image, use `--visualize=depth`
- optional `--depth-size` value which scales the size of the depth map relative to the input (the default is `1.0`)
- optional `--filter-mode` flag which selects `point` or `linear` filtering used for upsampling (the default is `linear`)
- optional `--colormap` flag which sets the color mapping to use during visualization (the default is `viridis_inverted`)

If you're using the [Docker container](aux-docker.md), it's recommended to save the output images to the `images/test` mounted directory.  These images will then be easily viewable from your host device under `jetson-inference/data/images/test` (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)). 

Here are some examples of mono depth estimation on indoor scenes:

``` bash
# C++
$ ./depthnet "images/room_*.jpg" images/test/depth_room_%i.jpg

# Python
$ ./depthnet.py "images/room_*.jpg" images/test/depth_room_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/depthnet-room-0.jpg">

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is then cached to disk, so future runs using the model will load faster.

And here are some taken from outdoor scenes:

``` bash
# C++
$ ./depthnet "images/trail_*.jpg" images/test/depth_trail_%i.jpg

# Python
$ ./depthnet.py "images/trail_*.jpg" images/test/depth_trail_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/depthnet-trail-0.jpg">

## Mono Depth from Video 

To run mono depth estimation on a live camera stream or video, pass in a device or file path from the [Camera Streaming and Multimedia](aux-streaming.md) page.

``` bash
# C++
$ ./depthnet /dev/video0     # csi://0 if using MIPI CSI camera

# Python
$ ./depthnet.py /dev/video0  # csi://0 if using MIPI CSI camera
```

<a href="https://www.youtube.com/watch?v=3_bU6Eqb4hE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/depthnet-video-0.jpg width="750"></a>

> **note**:  if the screen is too small to fit the output, you can use `--depth-scale=0.5` to reduce the size <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;of the depth image, or reduce the size of the camera with `--input-width=X --input-height=Y`


## Getting the Raw Depth Field

If you want to access the raw depth map, you can do so with `depthNet.GetDepthField()`.  This will return a single-channel floating point image that is typically smaller (224x224) than the original input - this represents the raw output of the model.  On the other hand, the colorized depth image used for visualization is upsampled to match the resolution of the original input (or whatever the `--depth-size` scale is set to).

Below is Python and C++ pseudocode for accessing the raw depth field:

#### Python

``` python
import jetson.inference
import jetson.utils

import numpy as np

# load mono depth network
net = jetson.inference.depthNet()

# depthNet re-uses the same memory for the depth field,
# so you only need to do this once (not every frame)
depth_field = net.GetDepthField()

# cudaToNumpy() will map the depth field cudaImage to numpy
# this mapping is persistent, so you only need to do it once
depth_numpy = jetson.utils.cudaToNumpy(depth_field)

print(f"depth field resolution is {depth_field.width}x{depth_field.height}, format={depth_field.format})

while True:
    img = input.Capture()	# assumes you have created an input videoSource stream
    net.Process(img)
    jetson.utils.cudaDeviceSynchronize() # wait for GPU to finish processing, so we can use the results on CPU
	
    # find the min/max values with numpy
    min_depth = np.amin(depth_numpy)
    max_depth = np.amax(depth_numpy)
```

#### C++

``` cpp
#include <jetson-inference/depthNet.h>

// load mono depth network
depthNet* net = depthNet::Create();

// depthNet re-uses the same memory for the depth field,
// so you only need to do this once (not every frame)
float* depth_field = net->GetDepthField();
const int depth_width = net->GetDepthWidth();
const int depth_height = net->GetDepthHeight();

while(true)
{
    uchar3* img = NUL;
    input->Capture(&img);  // assumes you have created an input videoSource stream
    net->Process(img, input->GetWidth(), input->GetHeight());

    // wait for the GPU to finish processing
    CUDA(cudaDeviceSynchronize()); 
	
    // you can now safely access depth_field from the CPU (or GPU)
    for( int y=0; y < depth_height; y++ )
        for( int x=0; x < depth_width; x++ )
	       printf("depth x=%i y=%i -> %f\n", x, y, depth_map[y * depth_width + x]);
}
```

Trying to measure absolute distances using mono depth can lead to inaccuracies as it's typically more effective at relative depth estimation.  The range of values in the raw depth field can vary depending on the scene, so these are often re-calculated dynamically.  For example, during visualization histogram equalization is performed on the depth field to distribute the colormap more evenly across the range of depth values.
	
Next, we're going to introduce the concepts of [Transfer Learning](pytorch-transfer-learning.md) and train our own DNN models on the Jetson using PyTorch.

##
<p align="right">Next | <b><a href="pytorch-transfer-learning.md">Transfer Learning with PyTorch</a></b>
<br/>
Back | <b><a href="backgroundnet.md">Background Removal</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
