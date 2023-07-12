<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="actionnet.md">Back</a> | <a href="depthnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Background Removal</sup></s></p>

# Background Removal
Background removal (aka background subtraction, or salient object detection) generates a mask that segments the foreground from the background of an image.  You can use it to replace or blur backgrounds (similar to video conferencing applications), or it clould aid in pre-processing for other vision DNN's like object detection/tracking or motion detection.  The model used is a fully-convolutional network [U²-Net](https://arxiv.org/abs/2005.09007).

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/backgroundnet-dog.jpg">

The [`backgroundNet`](../c/backgroundNet.h) object takes an image, and outputs the foreground mask.  [`backgroundNet`](../c/backgroundNet.h) can be used from [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#backgroundNet) and [C++](../c/backgroundNet.h).

As examples of using the `backgroundNet` class, there are sample programs for C++ and Python:

- [`backgroundnet.cpp`](../examples/backgroundnet/backgroundnet.cpp) (C++) 
- [`backgroundnet.py`](../python/examples/backgroundnet.py) (Python) 

## Running the Example

Here's an example of removing and replacing the background of an image:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/backgroundnet-bird.jpg">

``` bash
# C++
$ ./backgroundnet images/bird_0.jpg images/test/bird_mask.png                                 # remove the background (with alpha)
$ ./backgroundnet --replace=images/snow.jpg images/bird_0.jpg images/test/bird_replace.jpg    # replace the background

# Python
$ ./backgroundnet.py images/bird_0.jpg images/test/bird_mask.png                              # remove the background (with alpha)
$ ./backgroundnet.py --replace=images/snow.jpg images/bird_0.jpg images/test/bird_replace.jpg # replace the background
```

The `--replace` command-line argument accepts the filename of an image to replace the background with.  It will be re-scaled to the same resolution as the input.

### Live Streaming

To run background removal or replacement on a live camera stream, pass in a device from the [Camera Streaming and Multimedia](aux-streaming.md) page:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/backgroundnet-camera.jpg">

``` bash
# C++
$ ./backgroundnet /dev/video0                             # remove the background
$ ./backgroundnet --replace=images/coral.jpg /dev/video0  # replace the background

# Python
$ ./backgroundnet /dev/video0                             # remove the background
$ ./backgroundnet --replace=images/coral.jpg /dev/video0  # replace the background
```

By specifying an [output stream](aux-streaming.md#output-streams), you can view this on a display (the default), over the network (like with WebRTC), or save it to a video file.

##
<p align="right">Next | <b><a href="depthnet.md">Monocular Depth Estimation</a></b>
<br/>
Back | <b><a href="actionnet.md">Action Recognition</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
