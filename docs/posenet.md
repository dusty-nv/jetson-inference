<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-camera-2.md">Back</a> | <a href="depthnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Pose Estimation</sup></s></p>

# Pose Estimation
Pose estimation consists of locating various body parts (aka keypoints) forming a skeletal topology (aka links).  For this, we use models derived from the [OpenPose](https://arxiv.org/abs/1812.08008) DNN architecture that are capable of detecting the pose of multiple people per frame.  [Pre-trained models](#pre-trained-pose-estimation-models) are provided for human body and hand pose estimation.

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet.jpg" width="900">

The `poseNet` object accepts an image as input, and outputs a list of object poses.  Each object pose contains a list of detected keypoints, along with their locations and links between keypoints.  `poseNet` is available to use from [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h).

As examples of using the `poseNet` class, we provide sample programs for C++ and Python:

- [`posenet.cpp`](../examples/posenet/posenet.cpp) (C++) 
- [`posenet.py`](../python/examples/posenet.py) (Python) 

These samples are able to detect objects in images, videos, and camera feeds.  For more info about the various types of input/output streams supported, see the [Camera Streaming and Multimedia](aux-streaming.md) page.

### Pose Estimation on Images

First, let's try running the `posenet` sample on some examples images.  In addition to the input/output paths, there are some additional command-line options:

- optional `--network` flag which changes the [pose model](#pre-trained-pose-estimation-models) being used (the default is `resnet18-body`).
- optional `--overlay` flag which can be comma-separated combinations of `box`, `links`, `keypoints`, and `none`
	- The default is `--overlay=links,keypoints` which displays circles over they keypoints and lines over the links
- optional `--threshold` value which sets the minimum threshold for detection (the default is `0.15`).  

If you're using the [Docker container](aux-docker.md), it's recommended to save the output images to the `images/test` mounted directory.  These images will then be easily viewable from your host device under `jetson-inference/data/images/test` (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)). 

Here are some examples of human pose estimation with the default Pose-ResNet18-Body model:

``` bash
# C++
$ ./posenet "images/humans_*.jpg" images/test/humans_%i.jpg

# Python
$ ./posenet.py "images/humans_*.jpg" images/test/humans_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-peds-0.jpg" width="900">

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is then cached to disk, so future runs using the model will load faster.

There are also test images of people under `"images/peds_*.jpg"` that you can try as well.

### Pose Estimation on Video 

To run pose estimation on a live camera stream or videos, pass in a device or file from the [Camera Streaming and Multimedia](aux-streaming.md) page.

``` bash
# C++
$ ./posenet /dev/video0

# Python
$ ./posenet.py /dev/video0
```

<a href="https://www.youtube.com/watch?v=EbTyTJS9jOQ" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-pedestrians-video.jpg width="750"></a>

``` bash
# C++
$ ./posenet --network=resnet18-hand /dev/video0

# Python
$ ./posenet.py --network=resnet18-hand /dev/video0
```

<a href="https://www.youtube.com/watch?v=EbTyTJS9jOQ" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-pedestrians-video.jpg width="750"></a>

### Pre-trained Pose Estimation Models

Below are the pre-trained pose estimation networks available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `posenet` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Keypoints |
| ------------------------|--------------------|--------------------|-----------|
| Pose-ResNet18-Body      | `resnet18-body`    | `RESNET18_BODY`    | 18        |
| Pose-ResNet18-Hand      | `resnet18-hand`    | `RESNET18_HAND`    | 21        |
| Pose-DenseNet121-Body   | `densenet121-body` | `DENSENET121_BODY` | 18        |

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, Pose-ResNet18-Body is used if the optional `--network` flag isn't specified.


### Source Code

For reference, below is the source code to [`posenet.py`](../python/examples/posenet.py):

``` python
import jetson.inference
import jetson.utils

import argparse
import sys

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

# process frames until the user exits
while True:
    # capture the next image
    img = input.Capture()

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=opt.overlay)

    # print the pose results
    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

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

Next, we'll cover mono depth estimation.

##
<p align="right">Next | <b><a href="depthnet.md">Mono Depth Estimation</a></b>
<br/>
Back | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
