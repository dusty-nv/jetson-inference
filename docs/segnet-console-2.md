<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="detectnet-camera-2.md">Back</a> | <a href="segnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Semantic Segmentation with SegNet
The next deep learning capability we'll cover in this tutorial is **semantic segmentation**.  Semantic segmentation is based on image recognition, except the classifications occur at the pixel level as opposed to the entire image.  This is accomplished by *convolutionalizing* a pre-trained image recognition backbone, which transforms the model into a [Fully Convolutional Network (FCN)](https://arxiv.org/abs/1605.06211) capable of per-pixel labelling.  Especially useful for environmental perception, segmentation yields dense per-pixel classifications of many different potential objects per scene, including scene foregrounds and backgrounds.

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation.jpg" width="900">

`segNet` accepts as input the 2D image, and outputs a second image with the per-pixel classification mask overlay.  Each pixel of the mask corresponds to the class of object that was classified.  `segNet` is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/pytorch/docs/html/python/jetson.inference.html#segNet) and [C++](../c/segNet.h).  

As examples of using `segNet` we provide versions of a command-line interface for C++ and Python:

- [`segnet-console.cpp`](../examples/segnet-console/segnet-console.cpp) (C++) 
- [`segnet-console.py`](../python/examples/segnet-console.py) (Python) 

Later in the tutorial, we'll also cover segmentation on live camera streams from C++ and Python:

- [`segnet-camera.cpp`](../examples/segnet-camera/segnet-camera.cpp) (C++)
- [`segnet-camera.py`](../python/examples/segnet-camera.py) (Python) 

See [below](#pretrained-segmentation-models-available) for various pre-trained segmentation models available that use the FCN-ResNet18 network with realtime performance on Jetson.  Models are provided for a variety of environments and subject matter, including urban cities, off-road trails, and indoor office spaces and homes.

### Pre-Trained Segmentation Models Available

Below is a table of the pre-trained semantic segmentation models available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `segnet-console` used for loading them.  They're based on the 21-class FCN-ResNet18 network and have been trained on various datasets and resolutions using [PyTorch](https://github.com/dusty-nv/pytorch-segmentation), and were exported to [ONNX format](https://onnx.ai/) to be loaded with TensorRT.

| Dataset      | Resolution | CLI Argument | Accuracy | Jetson Nano | Jetson Xavier |
|:------------:|:----------:|--------------|:--------:|:-----------:|:-------------:|
| [Cityscapes](https://www.cityscapes-dataset.com/) | 512x256 | `fcn-resnet18-cityscapes-512x256` | 83.3% | 48 FPS | 480 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 1024x512 | `fcn-resnet18-cityscapes-1024x512` | 87.3% | 12 FPS | 175 FPS |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 2048x1024 | `fcn-resnet18-cityscapes-2048x1024` | 89.6% | 3 FPS | 47 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 576x320 | `fcn-resnet18-deepscene-576x320` | 96.4% | 26 FPS | 360 FPS |
| [DeepScene](http://deepscene.cs.uni-freiburg.de/) | 864x480 | `fcn-resnet18-deepscene-864x480` | 96.9% | 14 FPS | 190 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 512x320 | `fcn-resnet18-mhp-512x320` | 86.5% | 34 FPS | 370 FPS |
| [Multi-Human](https://lv-mhp.github.io/) | 640x360 | `fcn-resnet18-mhp-512x320` | 87.1% | 23 FPS | 325 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 320x320 | `fcn-resnet18-voc-320x320` | 85.9% | 45 FPS | 508 FPS |
| [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) | 512x320 | `fcn-resnet18-voc-512x320` | 88.5% | 34 FPS | 375 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 512x400 | `fcn-resnet18-sun-512x400` | 64.3% | 28 FPS | 340 FPS |
| [SUN RGB-D](http://rgbd.cs.princeton.edu/) | 640x512 | `fcn-resnet18-sun-640x512` | 65.1% | 17 FPS | 224 FPS |

* If the resolution is omitted from the CLI argument, the lowest resolution model is loaded
* Accuracy indicates the pixel classification accuracy across the model's validation dataset
* Performance is measured for GPU FP16 mode with JetPack 4.2.1, `nvpmodel 0` (MAX-N)

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>
<br/>

### Segmenting Images from the Command Line

The `segnet-console` program can be used to segment static images.  It accepts 3 command line parameters:

- the path to an input image  (`jpg, png, tga, bmp`)
- optional path to output image  (`jpg, png, tga, bmp`)
- optional `--network` flag changes the segmentation model being used (see [above](#pre-trained-segmentation-models-available))
- optional `--visualize` flag accepts `mask` or `overlay` modes (default is `overlay`)
- optional `--alpha` flag sets the alpha blending value for `overlay` (default is `120`)
- optional `--filter-mode` flag accepts `point` or `linear` sampling (default is `linear`)

Note that there are additional command line parameters available for loading custom models.  Launch the application with the `--help` flag to recieve more info about using them, or see the [`Code Examples`](../README.md#code-examples) readme.

Here are some example usages of the program:

#### C++

``` bash
$ ./segnet-console --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet-console --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet-console --network=<model> --visualize=mask input.jpg output.jpg # output the solid segmentation mask
```

#### Python

``` bash
$ ./segnet-console.py --network=<model> input.jpg output.jpg                  # overlay segmentation on original
$ ./segnet-console.py --network=<model> --alpha=200 input.jpg output.jpg      # make the overlay less opaque
$ ./segnet-console.py --network=<model> --visualize=mask input.jpg output.jpg # output the segmentation mask
```
<br/>

### Cityscapes

Let's look at some different scenarios.  Here's an example of segmenting an urban street scene with the [Cityscapes](https://www.cityscapes-dataset.com/) model:

``` bash
# C++
$ ./segnet-console --network=fcn-resnet18-cityscapes images/city_0.jpg output.jpg

# Python
$ ./segnet-console.py --network=fcn-resnet18-cityscapes images/city_0.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-city.jpg" width="900">

There are more test images called `city-*.jpg` found under the `images/` subdirectory for trying out the Cityscapes model.

### DeepScene

The [DeepScene dataset](http://deepscene.cs.uni-freiburg.de/) consists of off-road forest trails and vegetation, aiding in path-following for outdoor robots.  
Here's an example of generating the segmentation overlay and mask by specifying the `--visualize` argument:

#### C++
``` bash
$ ./segnet-console --network=fcn-resnet18-deepscene images/trail_0.jpg output_overlay.jpg                # overlay
$ ./segnet-console --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg output_mask.jpg  # mask
```

#### Python
``` bash
$ ./segnet-console.py --network=fcn-resnet18-deepscene images/trail_0.jpg output_overlay.jpg               # overlay
$ ./segnet-console.py --network=fcn-resnet18-deepscene --visualize=mask images/trail_0.jpg output_mask.jpg # mask
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-deepscene-0-overlay.jpg" width="850">
<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-deepscene-0-mask.jpg">

There are more sample images called `trail-*.jpg` located under the `images/` subdirectory.

### Multi-Human Parsing (MHP)

[Multi-Human Parsing](https://lv-mhp.github.io/) provides dense labeling of body parts, like arms, legs, head, and different types of clothing.  
 See the handful of test images named `humans-*.jpg` found under `images/` for trying out the MHP model:

``` bash
# C++
$ ./segnet-console --network=fcn-resnet18-mhp images/humans_0.jpg output.jpg

# Python
$ ./segnet-console.py --network=fcn-resnet18-mhp images/humans_0.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-0.jpg" width="825">
<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-1.jpg" width="825">

#### MHP Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-legend.jpg">

### Pascal VOC

[Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) is one of the original datasets used for semantic segmentation, containing various people, animals, vehicles, and household objects.  There are some sample images included named `object-*.jpg` for testing out the Pascal VOC model:

``` bash
# C++
$ ./segnet-console --network=fcn-resnet18-voc images/object_0.jpg output.jpg

# Python
$ ./segnet-console.py --network=fcn-resnet18-voc images/object_0.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-voc.jpg" width="900">

#### VOC Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-voc-legend.jpg">

### SUN RGB-D

The [SUN RGB-D](http://rgbd.cs.princeton.edu/) dataset provides segmentation ground-truth for many indoor objects and scenes commonly found in office spaces and homes.  See the images named `room-*.jpg` found under the `images/` subdirectory for testing out the SUN models:

``` bash
# C++
$ ./segnet-console --network=fcn-resnet18-sun images/room_0.jpg output.jpg

# Python
$ ./segnet-console.py --network=fcn-resnet18-sun images/room_0.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-sun.jpg" width="900">

#### SUN Classes

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-sun-legend.jpg">

### Processing a Directory of Images

For convenience, there's also a Python script provided called [`segnet-batch.py`](../python/examples/segnet-batch.py) for batch processing folders of images.

It's launched by specifying the `--network` option like above, and providing paths to the input and output directories:

``` bash
$ ./segnet-batch.py --network=<model> <input-dir> <output-dir>
```

That wraps up the segmentation models and command-line utilities.  Next, we'll run it on a live camera stream.

##
<p align="right">Next | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></b>
<br/>
Back | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
