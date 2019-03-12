<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-example.md">Back</a> | <a href="imagenet-training.md">Next</a> | </sup><a href="../README.md#training--inference"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Running the Live Camera Recognition Demo

Similar to the last example, the realtime image recognition demo is located in /aarch64/bin and is called [`imagenet-camera`](../imagenet-camera/imagenet-camera.cpp).
It runs on live camera stream and depending on user arguments, loads googlenet or alexnet with TensorRT. 
``` bash
$ ./imagenet-camera googlenet           # to run using googlenet
$ ./imagenet-camera alexnet             # to run using alexnet
```

The frames per second (FPS), classified object name from the video, and confidence of the classified object are printed to the openGL window title bar.  By default the application can recognize up to 1000 different types of objects, since Googlenet and Alexnet are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find included in the repo under [`data/networks/ilsvrc12_synset_words.txt`](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

> **note**:  by default, the Jetson's onboard CSI camera will be used as the video source.  If you wish to use a USB webcam instead, change the `DEFAULT_CAMERA` define at the top of [`imagenet-camera.cpp`](../imagenet-camera/imagenet-camera.cpp) to reflect the /dev/video V4L2 device of your USB camera.  The model it's tested with is Logitech C920. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange-camera.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple-camera.jpg" width="800">

Next, we will re-train the image recognition network on a customized dataset.

##
<p align="right">Next | <b><a href="imagenet-training.md">Re-Training the Recognition Network</a></b>
<br/>
Back | <b><a href="imagenet-example.md">Coding Your Own Image Recognition Program</a></b></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#training--inference"><sup>Table of Contents</sup></a></p>
