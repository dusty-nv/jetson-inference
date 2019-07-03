<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-camera-2.md">Back</a> | <a href="detectnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# Locating Objects with DetectNet
The previous image recognition examples output class probabilities representing the entire input image.   The second deep learning capability we're highlighting in this tutorial is **object detection**, and finding where in the frame various objects are located by extracting their bounding boxes.  Unlike image recognition, object detection networks are capable of detecting multiple independent objects per frame.

The `detectNet` object accepts an image as input, and outputs a list of coordinates of the detected bounding boxes along with their confidence values.  `detectNet` is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) and [C++](../detectNet.h).  See [below](#pretrained-detection-models-available) for various pre-trained detection models available for download. 

As examples of using `detectNet` we provide versions of a command-line interface for C++ and Python:

- [`detectnet-console.cpp`](../examples/detectnet-console/detectnet-console.cpp) (C++) 
- [`detectnet-console.py`](../python/examples/detectnet-console.py) (Python) 

Later in the tutorial, we'll also cover object detection on live camera streams from C++ and Python:

- [`detectnet-camera.cpp`](../examples/detectnet-camera/detectnet-camera.cpp) (C++)
- [`detectnet-camera.py`](../python/examples/detectnet-camera.py) (Python) 


### Detecting Objects from the Command Line

The `detectnet-console` program can be used to locate objects in static images.  It accepts 3 command line parameters:

- the path to an input image  (`jpg, png, tga, bmp`)
- optional path to output image  (`jpg, png, tga, bmp`)
- optional `--network` flag which changes the detection model being used (the default network is PedNet).  

Note that there are additional command line parameters available for loading custom models.  Launch the application with the `--help` flag to recieve more info about using them, or see the [`Code Examples`](../README.md#code-examples) readme.

Here's an example of locating humans in an image with the default PedNet model:

#### C++

``` bash
$ ./detectnet-console peds-004.jpg output.jpg
```

#### Python

``` bash
$ ./detectnet-console.py peds-004.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-00.jpg" width="900">


### Pretrained Detection Models Available

Below is a table of the pretrained object detection networks available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `detectnet-console` used for loading the pretrained model:

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
| SSD-Inception-v2        | `ssd-inception-v1` | `SSD_INCEPTION_V2` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
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


### Running Other Detection Models

You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, PedNet is loaded (pedestrian detection) if the optional `--network` flag isn't specified.

Let's try running some of the other COCO models:

``` bash
# C++
$ ./detectnet-console --network=coco-dog dog_1.jpg output_1.jpg

# Python
$ ./detectnet-console.py --network=coco-dog dog_1.jpg output_1.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-1.jpg)

``` bash
# C++
$ ./detectnet-console --network=coco-bottle bottle_0.jpg output_2.jpg

# Python
$ ./detectnet-console.py --network=coco-bottle bottle_0.jpg output_2.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-bottle-0.jpg)

``` bash
# C++
$ ./detectnet-console --network=coco-airplane airplane_0.jpg output_3.jpg 

# Python
$ ./detectnet-console.py --network=coco-airplane airplane_0.jpg output_3.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-airplane-0.jpg)


### Multi-class Object Detection Models

Some models support the detection of multiple types of objects.  For example, when using the `multiped` model on images containing luggage or baggage in addition to pedestrians, the 2nd object class is rendered with a green overlay:

``` bash
# C++
$ ./detectnet-console --network=multiped peds-003.jpg output_4.jpg

# Python
$ ./detectnet-console.py --network=multiped peds-003.jpg output_4.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-01.jpg" width="900">

Next, we'll run object detection on a live camera stream.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
