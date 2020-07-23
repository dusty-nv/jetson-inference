<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-console-2.md">Back</a> | <a href="detectnet-example-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Running the Live Camera Detection Demo

The [`detectnet.cpp`](../examples/detectnet/detectnet.cpp) / [`detectnet.py`](../python/examples/detectnet.py) sample that we used previously can also be used for realtime camera streaming.  The types of supported cameras include:

- MIPI CSI cameras (`csi://0`)
- V4L2 cameras (`/dev/video0`)
- RTP/RTSP streams (`rtsp://username:password@ip:port`)

For more information about video streams and protocols, please see the [Camera Streaming and Multimedia](aux-streaming.md) page.

Run the program with `--help` to see a full list of options - some of them specific to detectNet include:

- `--network` flag which changes the [detection model](detectnet-console-2.md#pre-trained-detection-models-available) being used (the default is SSD-Mobilenet-v2).
- `--overlay` flag which can be comma-separated combinations of `box`, `labels`, `conf`, and `none`
	- The default is `--overlay=box,labels,conf` which displays boxes, labels, and confidence values
- `--alpha` value which sets the alpha blending value used during overlay (the default is `120`).
- `--threshold` value which sets the minimum threshold for detection (the default is `0.5`).

Below are some typical scenarios for launching the program on a camera feed:

#### C++

``` bash
$ ./detectnet csi://0                    # MIPI CSI camera
$ ./detectnet /dev/video0                # V4L2 camera
$ ./detectnet /dev/video0 output.mp4     # save to video file
```

#### Python

``` bash
$ ./detectnet.py csi://0                 # MIPI CSI camera
$ ./detectnet.py /dev/video0             # V4L2 camera
$ ./detectnet.py /dev/video0 output.mp4  # save to video file
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Visualization

Displayed in the OpenGL window are the live camera stream overlayed with the bounding boxes of the detected objects.  Note that the SSD-based models currently have the highest performance.  Here is one using the `coco-dog` model:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-animals.jpg" width="800">

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-kitchen.jpg" width="800">

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/detectnet-ssd-laptops.jpg" width="800">

If the desired objects aren't being detected in the video feed or you're getting spurious detections, try decreasing or increasing the detection threshold with the `--threshold` parameter (the default is `0.5`).

Next, we'll cover creating the code for a camera detection app in Python.

##
<p align="right">Next | <b><a href="detectnet-example-2.md">Coding Your Own Object Detection Program</a></b>
<br/>
Back | <b><a href="detectnet-console-2.md">Detecting Objects from Images</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
