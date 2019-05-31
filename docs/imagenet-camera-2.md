<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-example-2.md">Back</a> | <a href="detectnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
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

To  flip the Camera you need to change the follwoing code:
```bash
$ cd ~/jetson-inference/utils/camera/
$ gedit gstCamera.cpp

// Find the follwing code:
	#if NV_TENSORRT_MAJOR > 1 && NV_TENSORRT_MAJOR < 5	// if JetPack 3.1-3.3 (different flip-method)
		const int flipMethod = 0;			// Xavier (w/TRT5) camera is mounted inverted
	#else
		const int flipMethod = 2;
	#endif	

//Change the code to:
	#if NV_TENSORRT_MAJOR > 1 && NV_TENSORRT_MAJOR < 5	// if JetPack 3.1-3.3 (different flip-method)
		const int flipMethod = 0;			// Xavier (w/TRT5) camera is mounted inverted
	#else
		const int flipMethod = 0;
	#endif	
//Save and close the code.

$ cd ~/jetson-inference/build/
$ make
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange-camera.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple-camera.jpg" width="800">

##
<p align="right">Next | <b><a href="detectnet-console-2.md">Locating Object Coordinates with DetectNet</a></b>
<br/>
Back | <b><a href="imagenet-example-2.md">Coding Your Own Image Recognition Program</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
