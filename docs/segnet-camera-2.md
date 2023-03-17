<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-console-2.md">Back</a> | <a href="posenet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Running the Live Camera Segmentation Demo
The [`segnet.cpp`](../examples/segnet/segnet.cpp) / [`segnet.py`](../python/examples/segnet.py) sample that we used previously can also be used for realtime camera streaming.  The types of supported cameras include:

- MIPI CSI cameras (`csi://0`)
- V4L2 cameras (`/dev/video0`)
- RTP/RTSP streams (`rtsp://username:password@ip:port`)

For more information about video streams and protocols, please see the [Camera Streaming and Multimedia](aux-streaming.md) page.

Run the program with `--help` to see a full list of options - some of them specific to segNet include:

- optional `--network` flag changes the segmentation model being used (see [available networks](segnet-console-2.md#pre-trained-segmentation-models-available))
- optional `--visualize` flag accepts `mask` and/or `overlay` modes (default is `overlay`)
- optional `--alpha` flag sets the alpha blending value for the overlay (default is `120`)
- optional `--filter-mode` flag accepts `point` or `linear` sampling (default is `linear`)

Below are some typical scenarios for launching the program - see [this table](segnet-console-2.md#pre-trained-segmentation-models-available) for the models available to use.

#### C++

``` bash
$ ./segnet --network=<model> csi://0                    # MIPI CSI camera
$ ./segnet --network=<model> /dev/video0                # V4L2 camera
$ ./segnet --network=<model> /dev/video0 output.mp4     # save to video file
```

#### Python

``` bash
$ ./segnet.py --network=<model> csi://0                 # MIPI CSI camera
$ ./segnet.py --network=<model> /dev/video0             # V4L2 camera
$ ./segnet.py --network=<model> /dev/video0 output.mp4  # save to video file
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Visualization

Displayed in the OpenGL window are the live camera stream overlayed with the segmentation output, alongside the solid segmentation mask for clarity.  Here are some examples of it being used with [different models](segnet-console-2.md#pre-trained-segmentation-models-available) that are available to try:

``` bash
# C++
$ ./segnet --network=fcn-resnet18-mhp csi://0

# Python
$ ./segnet.py --network=fcn-resnet18-mhp csi://0
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-mhp-camera.jpg" width="900">

``` bash
# C++
$ ./segnet --network=fcn-resnet18-sun csi://0

# Python
$ ./segnet.py --network=fcn-resnet18-sun csi://0
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-sun-camera.jpg" width="900">

``` bash
# C++
$ ./segnet --network=fcn-resnet18-deepscene csi://0

# Python
$ ./segnet.py --network=fcn-resnet18-deepscene csi://0
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-deepscene-camera.jpg" width="900">

Feel free to experiment with the different models and resolutions for indoor and outdoor environments.  


##
<p align="right">Next | <b><a href="posenet.md">Pose Estimation with PoseNet</a></b>
<br/>
Back | <b><a href="segnet-console-2.md">Segmenting Images from the Command Line</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
