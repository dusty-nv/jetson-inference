<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-console.md">Back</a> | <a href="segnet-dataset.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Running the Live Camera Detection Demo

Up next we have a realtime object detection camera demo available for C++ and Python:

- [`detectnet-camera.cpp`](../examples/detectnet-camera/detectnet-camera.cpp) (C++) 
- [`detectnet-camera.py`](../python/examples/detectnet-camera.py) (Python) 

Similar to the previous [`detectnet-console`](detectnet-console.md) example, these camera applications use detection networks, except that they process a live video feed from a camera.  `detectnet-camera` accepts 4 optional command-line parameters:

- `--network` flag setting the classification model (default is PedNet)
	- See [Pre-trained Detection Models Available](detectnet-console.md#pre-trained-detection-models-available) for the networks available to use.
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

Below are some typical scenarios for launching the program:

#### C++

``` bash
$ ./detectnet-camera                          # using PedNet,  default MIPI CSI camera (1280x720)
$ ./detectnet-camera --network=facenet        # using FaceNet, default MIPI CSI camera (1280x720)
$ ./detectnet-camera --camera=/dev/video0     # using PedNet,  V4L2 camera /dev/video0 (1280x720)
$ ./detectnet-camera --width=640 --height=480 # using PedNet,  default MIPI CSI camera (640x480)
```

#### Python

``` bash
$ ./detectnet-camera.py                          # using PedNet,  default MIPI CSI camera (1280x720)
$ ./detectnet-camera.py --network=facenet        # using FaceNet, default MIPI CSI camera (1280x720)
$ ./detectnet-camera.py --camera=/dev/video0     # using PedNet,  V4L2 camera /dev/video0 (1280x720)
$ ./detectnet-camera.py --width=640 --height=480 # using PedNet,  default MIPI CSI camera (640x480)
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Visualization

Displayed in the OpenGL window are the live camera stream overlayed with the bounding boxes of the detected objects.  Note that the SSD-based models currently have the highest performance.  Here is one using the `coco-dog` model:

``` bash
# C++
$ ./detectnet-camera --network=coco-dog

# Python
$ ./detectnet-camera.py --network=coco-dog
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet_camera_dog.jpg" width="800">

<br/>

##
<p align="right">Next | <b><a href="segnet-dataset.md">Semantic Segmentation with SegNet</a></b>
<br/>
Back | <b><a href="detectnet-console.md">Detecting Objects from the Command Line</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
