<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-tracking.md">Back</a> | <a href="segnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Semantic Segmentation with SegNet
The next deep learning capability we'll cover in this tutorial is **semantic segmentation**.  Semantic segmentation is based on image recognition, except the classifications occur at the pixel level as opposed to the entire image.  This is accomplished by *convolutionalizing* a pre-trained image recognition backbone, which transforms the model into a [Fully Convolutional Network (FCN)](https://arxiv.org/abs/1605.06211) capable of per-pixel labeling.  Especially useful for environmental perception, segmentation yields dense per-pixel classifications of many different potential objects per scene, including scene foregrounds and backgrounds.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation.jpg">

[`segNet`](../c/segNet.h) accepts as input the 2D image, and outputs a second image with the per-pixel classification mask overlay.  Each pixel of the mask corresponds to the class of object that was classified.  [`segNet`](../c/segNet.h) is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#segNet) and [C++](../c/segNet.h).  

As examples of using the `segNet` class, we provide sample programs C++ and Python:

- [`segnet.cpp`](../examples/segnet/segnet.cpp) (C++) 
- [`segnet.py`](../python/examples/segnet.py) (Python) 

These samples are able to segment images, videos, and camera feeds.  For more info about the various types of input/output streams supported, see the [Camera Streaming and Multimedia](aux-streaming.md) page.

See [below](#pretrained-segmentation-models-available) for various pre-trained segmentation models available that use the FCN-ResNet18 network with realtime performance on Jetson.  Models are provided for a variety of environments and subject matter, including urban cities, off-road trails, and indoor office spaces and homes.

### Pre-Trained Segmentation Models Available

Below is a table of the pre-trained semantic segmentation models available to use, and the associated `--network` argument to `segnet` used for loading them.  They're based on the 21-class FCN-ResNet18 network and have been trained on various datasets and resolutions using [PyTorch](https://github.com/dusty-nv/pytorch-segmentation), and were exported to [ONNX format](https://onnx.ai/) to be loaded with TensorRT.

| Dataset      | Resolution | CLI Argument | Accuracy | Jetson Nano | Jetson Xavier |
|:------------:|:----------:|--------------|:--------:|:-----------:|:-------------:|
| [Cityscapes](https://www.cityscapes-dataset.com/) | 512x256 | `fcn-resnet18-cityscapes-512x256` | 83.3% | 48 FPS | 480 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 1024x512 | `fcn-resnet18-cityscapes-1024x512` | 87.3% | 12 FPS | 175 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 2048x1024 | `fcn-resnet18-cityscapes-2048x1024` | 89.6% | 3 FPS | 47 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 576x320 | `fcn-resnet18-deepscene-576x320` | 96.4% | 26 FPS | 360 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 864x480 | `fcn-resnet18-deepscene-864x480` | 96.9% | 14 FPS | 190 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 512x320 | `fcn-resnet18-mhp-512x320` | 86.5% | 34 FPS | 370 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 640x360 | `fcn-resnet18-mhp-640x360` | 87.1% | 23 FPS | 325 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 320x320 | `fcn-resnet18-voc-320x320` | 85.9% | 45 FPS | 508 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 512x320 | `fcn-resnet18-voc-512x320` | 88.5% | 34 FPS | 375 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 512x400 | `fcn-resnet18-sun-512x400` | 64.3% | 28 FPS | 340 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 640x512 | `fcn-resnet18-sun-640x512` | 65.1% | 17 FPS | 224 FPS |

* If the resolution is omitted from the CLI argument, the lowest resolution model is loaded
* Accuracy indicates the pixel classification accuracy across the model's validation dataset
* Performance is measured for GPU FP16 mode with JetPack 4.2.1, `nvpmodel 0` (MAX-N)

### Segmenting Images from the Command Line

First, let's try using the `segnet` program to segment static images.  In addition to the input/output paths, there are some additional command-line options:

- optional `--network` flag changes the segmentation model being used (see [above](#pre-trained-segmentation-models-available))
- optional `--visualize` flag accepts `mask` and/or `overlay` modes (default is `overlay`)
- optional `--alpha` flag sets the alpha blending value for `overlay` (default is `120`)
- optional `--filter-mode` flag accepts `point` or `linear` sampling (default is `linear`)

Launch the application with the `--help` flag for more info, and refer to the [Camera Streaming and Multimedia](aux-streaming.md) page for supported input/output protocols.

Here are some example usages of the program:

#### C++

``` bash
$ ./segnet --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet --network=<model> --visualize=mask input.jpg output.jpg # output the solid segmentation mask
```

#### Python

``` bash
$ ./segnet.py --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet.py --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet.py --network=<model> --visualize=mask input.jpg output.jpg # output the segmentation mask
```

### Cityscapes

Let's look at some different scenarios.  Here's an example of segmenting an urban street scene with the [Cityscapes](https://www.cityscapes-dataset.com/) model:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-cityscapes images/city_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-city.jpg" width="1000">

There are more test images called `city-*.jpg` found under the `images/` subdirectory for trying out the Cityscapes model.

### DeepScene

The [DeepScene dataset](http://deepscene.cs.uni-freiburg.de/) consists of off-road forest trails and vegetation, aiding in path-following for outdoor robots.  
Here's an example of generating the segmentation overlay and mask by specifying the `--visualize` argument:

#### C++
``` bash
$ ./segnet --network=fcn-resnet18-deepscene images/trail_0.jpg images/test/output_overlay.jpg                # overlay
$ ./segnet --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg images/test/output_mask.jpg  # mask
```

#### Python
``` bash
$ ./segnet.py --network=fcn-resnet18-deepscene images/trail_0.jpg images/test/output_overlay.jpg               # overlay
$ ./segnet.py --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg images/test/output_mask.jpg # mask
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-deepscene-0-overlay.jpg" width="850">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-deepscene-0-mask.jpg">

There are more sample images called `trail-*.jpg` located under the `images/` subdirectory.

### Multi-Human Parsing (MHP)

[Multi-Human Parsing](https://lv-mhp.github.io/) provides dense labeling of body parts, like arms, legs, head, and different types of clothing.  
 See the handful of test images named `humans-*.jpg` found under `images/` for trying out the MHP model:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-mhp images/humans_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-mhp images/humans_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-mhp-0.jpg" width="825">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-mhp-1.jpg" width="825">

#### MHP Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-mhp-legend.jpg">

### Pascal VOC

[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is one of the original datasets used for semantic segmentation, containing various people, animals, vehicles, and household objects.  There are some sample images included named `object-*.jpg` for testing out the Pascal VOC model:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-voc images/object_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-voc images/object_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-voc.jpg" width="1000">

#### VOC Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-voc-legend.jpg">

### SUN RGB-D

The [SUN RGB-D](http://rgbd.cs.princeton.edu/) dataset provides segmentation ground-truth for many indoor objects and scenes commonly found in office spaces and homes.  See the images named `room-*.jpg` found under the `images/` subdirectory for testing out the SUN models:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-sun images/room_0.jpg images/test/output.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-sun images/room_0.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-sun.jpg" width="1000">

#### SUN Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-sun-legend.jpg">

### Processing a Directory or Sequence of Images

If you want to process a directory or sequence of images, you can launch the program with the path to the directory that contains images or a wildcard sequence:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-sun "images/room_*.jpg" images/test/room_output_%i.jpg

# Python
$ ./segnet.py --network=fcn-resnet18-sun "images/room_*.jpg" images/test/room_output_%i.jpg
```

> **note:** when using wildcards, always enclose it in quotes (`"*.jpg"`). Otherwise, the OS will auto-expand the sequence and modify the order of arguments on the command-line, which may result in one of the input images being overwritten by the output.

For more info about loading/saving sequences of images, see the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page.  Next, we'll run segmentation on a live camera or video stream.

##
<p align="right">Next | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></b>
<br/>
Back | <b><a href="detectnet-tracking.md">Object Tracking on Video</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
