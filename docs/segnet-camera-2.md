<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="segnet-console-2.md">Back</a> | <a href="pytorch-transfer-learning.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Running the Live Camera Segmentation Demo
Next we'll run realtime semantic segmentation on a live camera feed, available for C++ and Python:

- [`segnet-camera.cpp`](../examples/segnet-camera/segnet-camera.cpp) (C++)
- [`segnet-camera.py`](../python/examples/segnet-camera.py) (Python) 

Similar to the previous [`segnet-console`](segnet-console-2.md) example, these camera applications use segmentation networks, except that they process a live video feed instead.  `segnet-camera` accepts various **optional** command-line parameters, including:

- `--network` flag changes the segmentation model being used (see [available networks](segnet-console-2.md#pre-trained-segmentation-models-available))
- `--alpha` flag sets the alpha blending value for the overlay (default is `120`)
- `--filter-mode` flag accepts `point` or `linear` sampling (default is `linear`)
- `--camera` flag setting the camera device to use
	- MIPI CSI cameras are used by specifying the sensor index (`0` or `1`, ect.)
	- V4L2 USB cameras are used by specifying their `/dev/video` node (`/dev/video0`, `/dev/video1`, ect.)
	- The default is to use MIPI CSI sensor 0 (`--camera=0`)
- `--width` and `--height` flags setting the camera resolution (default is `1280x720`)
	- The resolution should be set to a format that the camera supports.
     - Query the available formats with the following commands:  
          ``` bash
          $ sudo apt-get install v4l-utils
          $ v4l2-ctl --list-formats-ext
          ```
You can combine the usage of these flags as needed, and there are additional command line parameters available for loading custom models.  Launch the application with the `--help` flag to recieve more info, or see the [`Examples`](../README.md#code-examples) readme.

Below are some typical scenarios for launching the program - see [this table](segnet-console-2.md#pre-trained-segmentation-models-available) for the models available to use.

#### C++

``` bash
$ ./segnet-camera --network=fcn-resnet18-mhp  # default MIPI CSI camera (1280x720)
$ ./segnet-camera --camera=/dev/video0        # V4L2 camera /dev/video0 (1280x720)
$ ./segnet-camera --width=640 --height=480    # default MIPI CSI camera (640x480)
```

#### Python

``` bash
$ ./segnet-camera.py --network=fcn-resnet18-mhp  # default MIPI CSI camera (1280x720)
$ ./segnet-camera.py --camera=/dev/video0        # V4L2 camera /dev/video0 (1280x720)
$ ./segnet-camera.py --width=640 --height=480    # default MIPI CSI camera (640x480)
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Visualization

Displayed in the OpenGL window are the live camera stream overlayed with the segmentation output, alongside the solid segmentation mask for clarity.  Here are some examples of it being used with [different models](segnet-console-2.md#pre-trained-segmentation-models-available) that are available to try:

``` bash
# C++
$ ./segnet-camera --network=fcn-resnet18-mhp

# Python
$ ./segnet-camera.py --network=fcn-resnet18-mhp
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-mhp-camera.jpg" width="900">

``` bash
# C++
$ ./segnet-camera --network=fcn-resnet18-sun

# Python
$ ./segnet-camera.py --network=fcn-resnet18-sun
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-sun-camera.jpg" width="900">

``` bash
# C++
$ ./segnet-camera --network=fcn-resnet18-deepscene

# Python
$ ./segnet-camera.py --network=fcn-resnet18-deepscene
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/pytorch/docs/images/segmentation-deepscene-camera.jpg" width="900">

Feel free to experiment with the different models and resolutions for indoor and outdoor environments.  Next, we're going to introduce the concepts of [Transfer Learning](pytorch-transfer-learning.md) and train some example DNN models on our Jetson using PyTorch.

##
<p align="right">Next | <b><a href="pytorch-transfer-learning.md">Transfer Learning with PyTorch</a></b>
<br/>
Back | <b><a href="segnet-console-2.md">Segmenting Images from the Command Line</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
