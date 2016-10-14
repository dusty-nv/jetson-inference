![Alt text](https://www.abaco.com/sites/default/files/default_images/product-strip.jpg)

# Abaco Systems Tx1 enable Small Form Factor solutions
## Deep Learning for Military Applications
This is a branch of the NVIDIA's deep learning inference workshop and end-to-end realtime object recognition library for Jetson TX1. Please reffer to the main branch for details and tips on setting up this demonstartion.

Included in this repo are resources for efficiently deploying neural networks using NVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)**.  

Vision primitives, such as [`imageNet`](imageNet.h) for image recognition and [`detectNet`](detectNet.h) for object localization, inherit from the shared [`tensorNet`](tensorNet.h) object.  Examples are provided for streaming from live camera feed and processing images from disk.  The actions to understand and apply these are represented as ten easy-to-follow steps.

## Support for USB webcam
On the Abaco Systems SFF box there is no CSI camera so [`gstCamera`](camera/gstCamera.h) has been modified to support the Logitech C920 Webcam. At the moment only imagenet-camera.cpp has been tested and proven to work with this modification. 

Modified pipelines can be supplied to the create function to support cameras using RGB colour space. The included pipeline in [`imagenet-camera.cpp`](imagenet-camera/imagenet-camera.cpp)  for the C920 is shown below:

    video/x-raw, width=(int)640, height=(int)480, format=RGB ! videoconvert ! video/x-raw, format=RGB ! videoconvert ! appsink name=mysink

## Goals
The aim of this project is to create a network trained on images that come from military applications such at Air / Sea / Land. Updating the network to work with this updated network for demonstration and as an example to a defence audiance. 

Please be patient whilst we build our DIGITS server to retrain the network and watch this space.
