<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">

# Change Log

Major updates and new features to this project will be listed in this document.

## April 8, 2022

* Added support for JetPack 5.0 and [Jetson AGX Orin](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit)
* Conditionally use NVIDIA V4L2-based hardware codecs when on JetPack 5.0 and newer
* Minor bug fixes and improvements

## August 3, 2021

* Added [Pose Estimation with PoseNet](docs/posenet.md) with pre-trained models
* Added [Mononocular Depth with DepthNet](docs/depthnet.md) with pre-trained models
* Added support for [`cudaMemcpy()` from Python](docs/aux-image.md#copying-images)
* Added support for [drawing 2D shapes with CUDA](docs/aux-image.md#drawing-shapes)

## August 31, 2020

* Added initial support for [Running in Docker Containers](docs/aux-docker.md)
* Changed OpenGL behavior to show window on first frame
* Minor bug fixes and improvements

## July 15, 2020

> **note:** API changes from this update are intended to be backwards-compatible, so previous code should still run.

* [Re-training SSD-Mobilenet](docs/pytorch-ssd.md) Object Detection tutorial with PyTorch
* Support for [collection of object detection datasets](docs/pytorch-collect-detection.md) and bounding-box labeling in `camera-capture` tool
* [`videoSource`](docs/aux-streaming.md#source-code) and [`videoOutput`](docs/aux-streaming.md#source-code) APIs for C++/Python that supports multiple types of video streams:
   * [MIPI CSI cameras](docs/aux-streaming.md#mipi-csi-cameras)
   * [V4L2 cameras](docs/aux-streaming.md#v4l2-cameras)
   * [RTP](docs/aux-streaming.md#rtp) / [RTSP](docs/aux-streaming.md#rtsp) 
   * [Videos](docs/aux-streaming.md#video-files) & [Images](docs/aux-streaming.md#image-files)
   * [Image sequences](docs/aux-streaming.md#image-files)
   * [OpenGL windows](docs/aux-streaming.md#output-streams)
* Unified the `-console` and `-camera` samples to process both images and video streams
   * [`imagenet.cpp`](examples/imagenet/imagenet.cpp) / [`imagenet.py`](python/examples/imagenet.py)
   * [`detectnet.cpp`](examples/detectnet/detectnet.cpp) / [`detectnet.py`](python/examples/detectnet.py)
   * [`segnet.cpp`](examples/segnet/segnet.cpp) / [`segnet.py`](python/examples/segnet.py)
* Support for `uchar3/uchar4/float3/float4` images (default is now `uchar3` as opposed to `float4`)
* Replaced opaque Python memory capsule with [`jetson.utils.cudaImage`](docs/aux-image.md#image-capsules-in-python) object
   * See [Image Capsules in Python](docs/aux-image.md#image-capsules-in-python) for more info
   * Images are now subscriptable/indexable from Python to directly access the pixel dataset
   * Numpy ndarray conversion now supports `uchar3/uchar4/float3/float4` formats
* [`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/blob/a587c20ad95d71efd47f9c91e3fbf703ad48644d/cuda/cudaColorspace.h#L31) automated colorspace conversion function (RGB, BGR, YUV, Bayer, grayscale, ect)
* Python CUDA bindings for `cudaResize()`, `cudaCrop()`, `cudaNormalize()`, `cudaOverlay()`
   * See [Image Manipulation with CUDA](docs/aux-image.md) and [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-examples.py) for examples of using these 
* Transitioned to using Python3 by default since Python 2.7 is now past EOL
* DIGITS tutorial is now marked as deprecated (replaced by PyTorch transfer learning tutorial)
* Logging can now be controlled/disabled from the command line (e.g. `--log-level=verbose`)

Thanks to everyone from the forums and GitHub who helped to test these updates in advance!

## October 3, 2019

* Added new pre-trained FCN-ResNet18 semantic segmentation models:

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

## July 19, 2019

* Python API support for imageNet, detectNet, and camera/display utilities</li>
* Python examples for processing static images and live camera streaming</li>
* Support for interacting with numpy ndarrays from CUDA</li>
* Onboard re-training of ResNet-18 models with PyTorch</li>
* Example datasets:  800MB Cat/Dog and 1.5GB PlantCLEF</li>
* Camera-based tool for collecting and labeling custom datasets</li>
* Text UI tool for selecting/downloading pre-trained models</li>
* New pre-trained image classification models (on 1000-class ImageNet ILSVRC)
   * ResNet-18, ResNet-50, ResNet-101, ResNet-152</li>
   * VGG-16, VGG-19</li>
   * Inception-v4</li>
* New pre-trained object detection models (on 90-class MS-COCO)
   * SSD-Mobilenet-v1</li>
   * SSD-Mobilenet-v2</li>
   * SSD-Inception-v2</li>
* API Reference documentation for C++ and Python</li>
   * Command line usage info for all examples, run with --help</li>
   * Output of network profiler times, including pre/post-processing</li>
   * Improved font rasterization using system TTF fonts</li>


##
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
