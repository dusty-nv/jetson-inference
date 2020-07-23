<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-example.md">Back</a> | <a href="imagenet-training.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Running the Live Camera Recognition Demo

Next we have a realtime image recognition camera demo available for C++ and Python:

- [`imagenet-camera.cpp`](../examples/imagenet-camera/imagenet-camera.cpp) (C++) 
- [`imagenet-camera.py`](../python/examples/imagenet-camera.py) (Python) 

Similar to the previous [`imagenet-console`](imagenet-console.md) example, the camera applications are built to the `/aarch64/bin` directory. They run on a live camera stream with OpenGL rendering and accept 4 optional command-line arguments:

- `--network` flag setting the classification model (default is GoogleNet)
	- See [Downloading Other Classification Models](imagenet-console.md#downloading-other-classification-models) for the networks available to use.
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
$ ./imagenet-camera                          # using GoogleNet, default MIPI CSI camera (1280x720)
$ ./imagenet-camera --network=resnet-18      # using ResNet-18, default MIPI CSI camera (1280x720)
$ ./imagenet-camera --camera=/dev/video0     # using GoogleNet, V4L2 camera /dev/video0 (1280x720)
$ ./imagenet-camera --width=640 --height=480 # using GoogleNet, default MIPI CSI camera (640x480)
```

#### Python

``` bash
$ ./imagenet-camera.py                          # using GoogleNet, default MIPI CSI camera (1280x720)
$ ./imagenet-camera.py --network=resnet-18      # using ResNet-18, default MIPI CSI camera (1280x720)
$ ./imagenet-camera.py --camera=/dev/video0     # using GoogleNet, V4L2 camera /dev/video0 (1280x720)
$ ./imagenet-camera.py --width=640 --height=480 # using GoogleNet, default MIPI CSI camera (640x480)
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

Displayed in the OpenGL window are the live camera stream, the classified object name, and the confidence of the classified object, along with the framerate of the network.  On Jetson Nano you should see up to around ~75 FPS for GoogleNet and ResNet-18 (faster on other Jetson's).

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_bear.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_camel.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/imagenet_camera_triceratops.jpg" width="800">

The application can recognize up to 1000 different types of objects, since the classification models are trained on the ILSVRC ImageNet dataset which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find in the repo under [`data/networks/ilsvrc12_synset_words.txt`](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

Next, we will re-train the image recognition network on a customized dataset.

##
<p align="right">Next | <b><a href="imagenet-training.md">Re-Training the Recognition Network</a></b>
<br/>
Back | <b><a href="imagenet-example.md">Coding Your Own Image Recognition Program</a></b></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
