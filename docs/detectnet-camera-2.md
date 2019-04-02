<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="detectnet-console-2.md">Back</a> | <a href="../README.md#hello-ai-world-inference-only">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Running the Live Camera Detection Demo

Similar to the previous example, [`detectnet-camera`](../detectnet-camera/detectnet-camera.cpp) runs the object detection networks on live video feed from the Jetson onboard camera.  Launch it from command line along with the type of desired network:

``` bash
$ ./detectnet-camera facenet        # run using facial recognition network
$ ./detectnet-camera multiped       # run using multi-class pedestrian/luggage detector
$ ./detectnet-camera pednet         # run using original single-class pedestrian detector
$ ./detectnet-camera coco-bottle    # detect bottles/soda cans in the camera
$ ./detectnet-camera coco-dog       # detect dogs in the camera
$ ./detectnet-camera                # by default, program will run using multiped
```

> **note**:  to achieve maximum performance while running detectnet, increase the Jetson clock limits by running the script:
>  `sudo ~/jetson_clocks.sh`

<br/>

> **note**:  by default, the Jetson's onboard CSI camera will be used as the video source.  If you wish to use a USB webcam instead, change the `DEFAULT_CAMERA` define at the top of [`detectnet-camera.cpp`](../detectnet-camera/detectnet-camera.cpp) to reflect the /dev/video V4L2 device of your USB camera and recompile.  The webcam model it's tested with is Logitech C920.  

<br/>

## What's Next

This is the last step of the *Hello AI World* tutorial, which covers inferencing on Jetson with TensorRT.  

To recap, together we've covered:

* Using image recognition networks to classify images
* Coding your own image recognition program in C++
* Classifying video from a live camera stream
* Performing object detection to locate object coordinates

Next, we encourage you to follow our full **[Training + Inference](https://github.com/dusty-nv/jetson-inference#two-days-to-a-demo-training--inference)** tutorial, which also covers the re-training of these networks on custom datasets.  This way, you can collect your own data and have the models recognize objects specific to your applications.  The full tutorial also covers semantic segmentation, which is like image classification, but on a per-pixel level instead of predicting one class for the entire image.  Good luck!

##
<p align="right">Back | <b><a href="detectnet-console-2.md">Detecting Objects from the Command Line</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
