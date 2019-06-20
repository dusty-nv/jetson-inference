<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-example-2.md">Back</a> | <a href="detectnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Running the Live Camera Recognition Demo

Next we have a realtime image recognition camera demo available for C++ and Python:

- [`imagenet-camera.cpp`](../examples/imagenet-camera/imagenet-camera.cpp) (C++) 
- [`imagenet-camera.py`](../python/examples/imagenet-camera.py) (Python) 

Similar to the previous [`imagenet-console`](imagenet-console-2.md) example, the camera applications are built to the `/aarch64/bin` directory. It runs on a live camera stream with OpenGL rendering and accepts 4 optional command-line arguments:

- optional `--network` flag specifying the classification model (default is GoogleNet)
	- See [Downloading Other Classification Models](imagenet-console-2.md#downloading-other-classification-models] for the networks available to use
- optional `--camera` flag specifying the camera device to use
	- MIPI CSI cameras are used by specifying the sensor index (i.e. `0` or `1`, ect.)
	- V4L2 USB cameras are used by specifying their `/dev/video` node (i.e. `/dev/video0`)
	- The default is to use MIPI CSI sensor 0 (`--camera=0`)
- optional `--width` and `--height` flags specifying the camera resolution (default is `1280x720`)

> **note**:  for example cameras to use, see these section of the Jetson wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:  [`https://elinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  (comes with onboard MIPI CSI sensor module (0V5693)<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>

Below are examples of launching the camera program in C++ or Python:

#### C++

``` bash
$ ./imagenet-camera                          # run using GoogleNet, default MIPI CSI camera (1280x720)
$ ./imagenet-camera --network=resnet-18      # run using ResNet-18, default MIPI CSI camera (1280x720)
$ ./imagenet-camera --camera=/dev/video0     # run using GoogleNet, V4L2 camera /dev/video0 (1280x720)
$ ./imagenet-camera --width=640 --height=480 # run using GoogleNet, default MIPI CSI camera (640x480)
```

#### Python

``` bash
$ ./imagenet-camera.py                          # run using GoogleNet, default MIPI CSI camera (1280x720)
$ ./imagenet-camera.py --network=resnet-18      # run using ResNet-18, default MIPI CSI camera (1280x720)
$ ./imagenet-camera.py --camera=/dev/video0     # run using GoogleNet, V4L2 camera /dev/video0 (1280x720)
$ ./imagenet-camera.py --width=640 --height=480 # run using GoogleNet, default MIPI CSI camera (640x480)
```

Note that you can combine these command line flags above as needed.  Displayed in the OpenGL window are the camera stream, the classified object name, and the confidence of the classified object, along with the framerate of the network.  On Jetson Nano you should see up to around 75 FPS for GoogleNet and ResNet-18 (faster on other Jetson's).

By default the application can recognize up to 1000 different types of objects, since the classification models are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find included in the repo under [`data/networks/ilsvrc12_synset_words.txt`](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)


<img src="https://github.com/dusty-nv/jetson-inference/python/master/docs/images/imagenet_camera_bear.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet_camera_camel.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet_camera_triceratops.jpg" width="800">

This concludes the section of Hello AI World on Image Recognition.  Next, we're going to start using Object Detection networks, which provide us with the bounding box coordinates of multiple objects per frame.

##
<p align="right">Next | <b><a href="detectnet-console-2.md">Locating Object Coordinates with DetectNet</a></b>
<br/>
Back | <b><a href="imagenet-example-2.md">Coding Your Own Image Recognition Program</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
