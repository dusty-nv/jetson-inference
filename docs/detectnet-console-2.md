<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-tagging.md">Back</a> | <a href="detectnet-camera-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# Locating Objects with DetectNet
The previous recognition examples output class probabilities representing the entire input image.  Next we're going to focus on **object detection**, and finding where in the frame various objects are located by extracting their bounding boxes.  Unlike image classification, object detection networks are capable of detecting many different objects per frame.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet.jpg" >

The [`detectNet`](../c/detectNet.h) object accepts an image as input, and outputs a list of coordinates of the detected bounding boxes along with their classes and confidence values.  [`detectNet`](../c/detectNet.h) is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet) and [C++](../c/detectNet.h).  See below for various [pre-trained detection models](#pre-trained-detection-models-available)  available for download.  The default model used is a [91-class](../data/networks/ssd_coco_labels.txt) SSD-Mobilenet-v2 model trained on the MS COCO dataset, which achieves realtime inferencing performance on Jetson with TensorRT. 

As examples of using the `detectNet` class, we provide sample programs for C++ and Python:

- [`detectnet.cpp`](../examples/detectnet/detectnet.cpp) (C++) 
- [`detectnet.py`](../python/examples/detectnet.py) (Python) 

These samples are able to detect objects in images, videos, and camera feeds.  For more info about the various types of input/output streams supported, see the [Camera Streaming and Multimedia](aux-streaming.md) page.

### Detecting Objects from Images

First, let's try using the `detectnet` program to locates objects in static images.  In addition to the input/output paths, there are some additional command-line options:

- optional `--network` flag which changes the [detection model](detectnet-console-2.md#pre-trained-detection-models-available) being used (the default is SSD-Mobilenet-v2).
- optional `--overlay` flag which can be comma-separated combinations of `box`, `lines`, `labels`, `conf`, and `none`
	- The default is `--overlay=box,labels,conf` which displays boxes, labels, and confidence values
	- The `box` option draws filled bounding boxes, while `lines` draws just the unfilled outlines
- optional `--alpha` value which sets the alpha blending value used during overlay (the default is `120`).
- optional `--threshold` value which sets the minimum threshold for detection (the default is `0.5`).  

If you're using the [Docker container](aux-docker.md), it's recommended to save the output images to the `images/test` mounted directory.  These images will then be easily viewable from your host device under `jetson-inference/data/images/test` (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)). 

Here are some examples of detecting pedestrians in images with the default SSD-Mobilenet-v2 model:

``` bash
# C++
$ ./detectnet --network=ssd-mobilenet-v2 images/peds_0.jpg images/test/output.jpg     # --network flag is optional

# Python
$ ./detectnet.py --network=ssd-mobilenet-v2 images/peds_0.jpg images/test/output.jpg  # --network flag is optional
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-ssd-peds-0.jpg" >

``` bash
# C++
$ ./detectnet images/peds_1.jpg images/test/output.jpg

# Python
$ ./detectnet.py images/peds_1.jpg images/test/output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-ssd-peds-1.jpg" >

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is then cached to disk, so future runs using the model will load faster.

Below are more detection examples output from the console programs.  The [91-class](../data/networks/ssd_coco_labels.txt) MS COCO dataset that the SSD-based models were trained on include people, vehicles, animals, and assorted types of household objects to detect.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-animals.jpg" >

Various images are found under `images/` for testing, such as `cat_*.jpg`, `dog_*.jpg`, `horse_*.jpg`, `peds_*.jpg`, ect. 

### Processing a Directory or Sequence of Images

If you have multiple images that you'd like to process at one time, you can launch the `detectnet` program with the path to a directory that contains images or a wildcard sequence:

```bash
# C++
./detectnet "images/peds_*.jpg" images/test/peds_output_%i.jpg

# Python
./detectnet.py "images/peds_*.jpg" images/test/peds_output_%i.jpg
```

> **note:** when using wildcards, always enclose it in quotes (`"*.jpg"`). Otherwise, the OS will auto-expand the sequence and modify the order of arguments on the command-line, which may result in one of the input images being overwritten by the output.

For more info about loading/saving sequences of images, see the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page.

### Processing Video Files

You can also process videos from disk.  For more info about loading/saving videos, see [here](aux-streaming.md#video-files).

``` bash
# Download test video
wget https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4 -O pedestrians.mp4

# C++
./detectnet pedestrians.mp4 images/test/pedestrians_ssd.mp4

# Python
./detectnet.py pedestrians.mp4 images/test/pedestrians_ssd.mp4
```

<a href="https://www.youtube.com/watch?v=EbTyTJS9jOQ" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-ssd-pedestrians-video.jpg width="750"></a>

``` bash
# Download test video
wget https://nvidia.box.com/shared/static/i5i81mkd9wdh4j7wx04th961zks0lfh9.avi -O parking.avi

# C++
./detectnet parking.avi images/test/parking_ssd.avi

# Python
./detectnet.py parking.avi images/test/parking_ssd.avi
```

<a href="https://www.youtube.com/watch?v=iB86W-kloPE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-ssd-parking-video.jpg width="585"></a>

Remember that you can use the `--threshold` setting to change the detection sensitivity up or down (the default is 0.5).

### Pre-trained Detection Models Available

Below is a table of the pre-trained object detection networks available to use, and the associated `--network` argument to `detectnet` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| SSD-Inception-v2        | `ssd-inception-v2` | `SSD_INCEPTION_V2` | 91 ([COCO classes](../data/networks/ssd_coco_labels.txt))     |
| TAO PeopleNet           | `peoplenet`        | `PEOPLENET`        | person, bag, face    |
| TAO PeopleNet (pruned)  | `peoplenet-pruned` | `PEOPLENET_PRUNED` | person, bag, face    |
| TAO DashCamNet          | `dashcamnet`       | `DASHCAMNET`       | person, car, bike, sign |
| TAO TrafficCamNet       | `trafficcamnet`    | `TRAFFICCAMNET`    | person, car, bike, sign | 
| TAO FaceDetect          | `facedetect`       | `FACEDETECT`       | face                 |

<details>
<summary>Legacy Detection Models</summary>

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
| facenet-120             | `facenet`          | `FACENET`          | faces                |

</details>

### Running Different Detection Models

You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, SSD-Mobilenet-v2 if the optional `--network` flag isn't specified.

For example, if you chose to download SSD-Inception-v2 with the [Model Downloader](building-repo-2.md#downloading-models) tool, you can use it like so:

``` bash
# C++
$ ./detectnet --network=ssd-inception-v2 input.jpg output.jpg

# Python
$ ./detectnet.py --network=ssd-inception-v2 input.jpg output.jpg
```

### Source Code

For reference, below is the source code to [`detectnet.py`](../python/examples/detectnet.py):

``` python
import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.")

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# process frames until the user exits
while True:
	# capture the next image
	img = input.Capture()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, overlay=opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)

	# render the image
	output.Render(img)

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break
```

Next, we'll run object detection on a live camera stream.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-tagging.md">Multi-Label Classification for Image Tagging</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
