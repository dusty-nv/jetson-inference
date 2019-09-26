<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-camera-2.md">Back</a> | <a href="detectnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# Locating Objects with DetectNet
The previous recognition examples output class probabilities representing the entire input image.  Next we're going to focus on **object detection**, and finding where in the frame various objects are located by extracting their bounding boxes.  Unlike image classification, object detection networks are capable of detecting many different objects per frame.

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/detectnet.jpg" width="900">

The `detectNet` object accepts an image as input, and outputs a list of coordinates of the detected bounding boxes along with their classes and confidence values.  `detectNet` is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) and [C++](../c/detectNet.h).  See below for various [pre-trained detection models](#pretrained-detection-models-available)  available for download.  The default model used is a [91-class](../data/networks/ssd_coco_labels.txt) SSD-Mobilenet-v2 model trained on the MS COCO dataset, which achieves realtime inferencing performance on Jetson with TensorRT. 

As examples of using `detectNet` we provide versions of a command-line interface for C++ and Python:

- [`detectnet-console.cpp`](../examples/detectnet-console/detectnet-console.cpp) (C++) 
- [`detectnet-console.py`](../python/examples/detectnet-console.py) (Python) 

Later in the tutorial, we'll also cover object detection on live camera streams from C++ and Python:

- [`detectnet-camera.cpp`](../examples/detectnet-camera/detectnet-camera.cpp) (C++)
- [`detectnet-camera.py`](../python/examples/detectnet-camera.py) (Python) 


### Detecting Objects from the Command Line

The `detectnet-console` program locates objects in static images.  Some of it's important command line parameters are:

- the path to an input image  (`jpg, png, tga, bmp`)
- optional path to output image  (`jpg, png, tga, bmp`)
- optional `--network` flag which changes the [detection model](detectnet-console-2.md#pre-trained-detection-models-available) being used (the default is SSD-Mobilenet-v2).
- optional `--overlay` flag which can be comma-separated combinations of `box`, `labels`, `conf`, and `none`
	- The default is `--overlay=box,labels,conf` which displays boxes, labels, and confidence values
- optional `--alpha` value which sets the alpha blending value used during overlay (the default is `120`).
- optional `--threshold` value which sets the minimum threshold for detection (the default is `0.5`).  

Note that there are additional command line parameters available for loading custom models.  Launch the application with the `--help` flag to recieve more info about using them, or see the [`Code Examples`](../README.md#code-examples) readme.

Here are some examples of detecting pedestrians in images with the default SSD-Mobilenet-v2 model:

``` bash
# C++
$ ./detectnet-console --network=ssd-mobilenet-v2 images/peds_0.jpg output.jpg     # --network flag is optional

# Python
$ ./detectnet-console.py --network=ssd-mobilenet-v2 images/peds_0.jpg output.jpg  # --network flag is optional
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/detectnet-ssd-peds-0.jpg" width="900">

``` bash
# C++
$ ./detectnet-console images/peds_1.jpg output.jpg

# Python
$ ./detectnet-console.py images/peds_1.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/detectnet-ssd-peds-1.jpg" width="900">

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file then cached to disk, so future runs using the model will load faster.

Below are more detection examples output from the console programs.  The [91-class](../data/networks/ssd_coco_labels.txt) MS COCO dataset that the SSD-based models were trained on include people, vehicles, animals, and assorted types of household objects to detect.

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/detectnet-animals.jpg" width="900">

Various images are found under `images/` for testing, such as `cat_*.jpg`, `dog_*.jpg`, `horse_*.jpg`, `peds_*.jpg`, ect. 


### Pre-trained Detection Models Available

Below is a table of the pre-trained object detection networks available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `detectnet-console` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Inception-v2        | `ssd-inception-v1` | `SSD_INCEPTION_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
| facenet-120             | `facenet`          | `FACENET`          | faces                |

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


### Running Different Detection Models

You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, SSD-Mobilenet-v2 if the optional `--network` flag isn't specified.

For example, if you chose to download SSD-Inception-v2 with the [Model Downloader](building-repo-2.md#downloading-models) tool, you can use it like so:

``` bash
# C++
$ ./detectnet-console --network=ssd-inception-v2 input.jpg output.jpg

# Python
$ ./detectnet-console.py --network=ssd-inception-v2 input.jpg output.jpg
```

### Source Code

For reference, below is the source code to [`detectnet-console.py`](../python/examples/detectnet-console.py):

``` python
import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in an image using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("file_in", type=str, help="filename of the input image to process")
parser.add_argument("file_out", type=str, help="filename of the output image to save")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use")

opt = parser.parse_known_args()[0]

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.file_in)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# detect objects in the image (with overlay)
detections = net.Detect(img, width, height, opt.overlay)

# print the detections
print("detected {:d} objects in image".format(len(detections)))

for detection in detections:
	print(detection)

# print out timing info
net.PrintProfilerTimes()

# save the output image with the bounding box overlays
jetson.utils.saveImageRGBA(opt.file_out, img, width, height)
```

Next, we'll run object detection on a live camera stream.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
