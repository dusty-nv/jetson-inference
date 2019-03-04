<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Deploying Deep Learning
Welcome to our training guide for inference and [deep vision](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html) runtime library for NVIDIA **[DIGITS](https://github.com/NVIDIA/DIGITS)** and **[Jetson Xavier/TX1/TX2](http://www.nvidia.com/object/embedded-systems.html)**.

This repo uses NVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)** for efficiently deploying neural networks onto the embedded platform, improving performance and power efficiency using graph optimizations, kernel fusion, and half-precision FP16 on the Jetson.

Vision primitives, such as [`imageNet`](imageNet.h) for image recognition, [`detectNet`](detectNet.h) for object localization, and [`segNet`](segNet.h) for segmentation, inherit from the shared [`tensorNet`](tensorNet.h) object.  Examples are provided for streaming from live camera feed and processing images from disk.  See the **[Deep Vision API Reference Specification](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html)** for accompanying documentation. 

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-primitives.png" width="800">

> &gt; &nbsp; See **[Image Segmentation](#image-segmentation-with-segnet)** models and training guide with aerial drone dataset. <br/>
> &gt; &nbsp; Try **[Object Detection](#locating-object-coordinates-using-detectnet)** training guide using DIGITS & MS-COCO training dataset. <br/>
> &gt; &nbsp; Use **[Image Recognition](#re-training-the-network-with-digits)** training guide with DIGITS & ImageNet ILSVRC12 dataset. <br/>
> &gt; &nbsp; Check out our next GitHub tutorial, **[Deep Reinforcement Learning in Robotics](http://github.com/dusty-nv/jetson-reinforcement)**. <br/>
> &gt; &nbsp; Jetson AGX Xavier Developer Kit and JetPack 4.1.1 DP is **[now available](https://news.developer.nvidia.com/nvidia-jetson-agx_xavier-developer-kit-now-available/)** and supported in the repo.

### **Table of Contents**

* [DIGITS Workflow](#digits-workflow) 
* [System Setup](#system-setup)
* [Building from Source on Jetson](#building-from-source-on-jetson)
* [Classifying Images with ImageNet](#classifying-images-with-imagenet)
	* [Using the Console Program on Jetson](#using-the-console-program-on-jetson)
	* [Running the Live Camera Recognition Demo](#running-the-live-camera-recognition-demo)
	* [Re-training the Network with DIGITS](#re-training-the-network-with-digits)
	* [Downloading Image Recognition Dataset](#downloading-image-recognition-dataset)
	* [Customizing the Object Classes](#customizing-the-object-classes)
	* [Importing Classification Dataset into DIGITS](#importing-classification-dataset-into-digits)
	* [Creating Image Classification Model with DIGITS](#creating-image-classification-model-with-digits)
	* [Testing Classification Model in DIGITS](#testing-classification-model-in-digits)
	* [Downloading Model Snapshot to Jetson](#downloading-model-snapshot-to-jetson)
	* [Loading Custom Models on Jetson](#loading-custom-models-on-jetson)
* [Locating Object Coordinates using DetectNet](#locating-object-coordinates-using-detectnet)
	* [Detection Data Formatting in DIGITS](#detection-data-formatting-in-digits)
	* [Downloading the Detection Dataset](#downloading-the-detection-dataset)
	* [Importing the Detection Dataset into DIGITS](#importing-the-detection-dataset-into-digits)
	* [Creating DetectNet Model with DIGITS](#creating-detectnet-model-with-digits)
	* [Testing DetectNet Model Inference in DIGITS](#testing-detectnet-model-inference-in-digits)
	* [Downloading the Model Snapshot to Jetson](#downloading-the-model-snapshot-to-jetson)
	* [DetectNet Patches for TensorRT](#detectnet-patches-for-tensorrt)
	* [Processing Images from the Command Line on Jetson](#processing-images-from-the-command-line-on-jetson)
	* [Multi-class Object Detection Models](#multi-class-object-detection-models)
	* [Running the Live Camera Detection Demo on Jetson](#running-the-live-camera-detection-demo-on-jetson)
* [Image Segmentation with SegNet](#image-segmentation-with-segnet)
	* [Downloading Aerial Drone Dataset](#downloading-aerial-drone-dataset)
	* [Importing the Aerial Dataset into DIGITS](#importing-the-aerial-dataset-into-digits)
	* [Generating Pretrained FCN-Alexnet](#generating-pretrained-fcn-alexnet)
	* [Training FCN-Alexnet with DIGITS](#training-fcn-alexnet-with-digits)
	* [Testing Inference Model in DIGITS](#testing-inference-model-in-digits)
	* [FCN-Alexnet Patches for TensorRT](#fcn-alexnet-patches-for-tensorrt)
	* [Running Segmentation Models on Jetson](#running-segmentation-models-on-jetson)

**Recommended System Requirements**

Training GPU:  Maxwell, Pascal, or Volta-based GPU (ideally with at least 6GB video memory)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;optionally, AWS P2/P3 instance or Microsoft Azure N-series  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 14.04 x86_64 or Ubuntu 16.04 x86_64.

Deployment:    &nbsp;&nbsp;Jetson Xavier Developer Kit with JetPack 4.0 or newer (Ubuntu 18.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX2 Developer Kit with JetPack 3.0 or newer (Ubuntu 16.04 aarch64).  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX1 Developer Kit with JetPack 2.3 or newer (Ubuntu 16.04 aarch64).

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against the following BSP versions for Jetson AGX Xavier and Jetson TX1/TX2: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.1.1 DP / L4T R31.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 GA<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.1 DP EA / L4T R31.0.2 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 RC<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson AGX Xavier - JetPack 4.0 DP EA / L4T R31.0.1 aarch64 (Ubuntu 18.04 LTS) inc. TensorRT 5.0 RC<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.3 / L4T R28.2.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 4.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.3 / L4T R28.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 4.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.2 / L4T R28.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 2.1<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 2.1<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 1.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 1.0<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)

Note that TensorRT samples from the repo are intended for deployment onboard Jetson, however when cuDNN and TensorRT have been installed on the host side, the TensorRT samples in the repo can be compiled for PC.

## Extra Resources

In this area, links and resources for deep learning developers are listed:

* [Appendix](docs/aux-contents.md)
	* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes
     * [NVIDIA AI IoT](https://github.com/NVIDIA-AI-IOT) - NVIDIA Jetson GitHub repositories
     * [Jetson eLinux Wiki](https://www.eLinux.org/Jetson) - Jetson eLinux Wiki


## Legacy Links

<details open>
<summary>Since the documentation has been re-organized, below are links mapping the previous content to the new locations.</summary>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(click on the arrow above to hide this section)

### DIGITS Workflow

See [DIGITS Workflow](docs/digits-workflow.md)

### System Setup

See [DIGITS Setup](docs/digits-setup.md)

#### Running JetPack on the Host

See [JetPack Setup](docs/jetpack-setup.md)

#### Installing Ubuntu on the Host

See [DIGITS Setup](docs/digits-setup.md#installing-ubuntu-on-the-host)

#### Setting up host training PC with NGC container	

See [DIGITS Setup](docs/digits-setup.md#setting-up-host-training-pc-with-ngc-container)

#### Installing the NVIDIA driver

See [DIGITS Setup](docs/digits-setup.md#installing-the-nvidia-driver)

#### Installing Docker

See [DIGITS Setup](docs/digits-setup.md#installing-docker)

#### NGC Sign-up 

See [DIGITS Setup](docs/digits-setup.md#ngc-sign-up)

#### Setting up data and job directories

See [DIGITS Setup](docs/digits-setup.md#setting-up-data-and-job-directories)

#### Starting DIGITS container

See [DIGITS Setup](docs/digits-setup.md#starting-digits-container)

#### Natively setting up DIGITS on the Host 

See [DIGITS Native Setup](docs/digits-native.md)

#### Installing NVIDIA Driver on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-nvidia-driver-on-the-host)

#### Installing cuDNN on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-cudnn-on-the-host)

#### Installing NVcaffe on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-nvcaffe-on-the-host)

#### Installing DIGITS on the Host

See [DIGITS Native Setup](docs/digits-native.md#installing-digits-on-the-host)

#### Starting the DIGITS Server

See [DIGITS Native Setup](docs/digits-native.md#starting-the-digits-server)

### Building from Source on Jetson

See [Building the Repo from Source](docs/building-repo.md)
      
#### Cloning the Repo

See [Building the Repo from Source](docs/building-repo.md#cloning-the-repo)

#### Configuring with CMake

See [Building the Repo from Source](docs/building-repo.md#configuring-with-cmake)

#### Compiling the Project

See [Building the Repo from Source](docs/building-repo.md#compiling-the-project)

#### Digging Into the Code

See [Building the Repo from Source](docs/building-repo.md#digging-into-the-code)

### Classifying Images with ImageNet

See [Classifying Images with ImageNet](docs/imagenet-console.md)

#### Using the Console Program on Jetson

See [Classifying Images with ImageNet](docs/imagenet-console.md#using-the-console-program-on-jetson)

### Running the Live Camera Recognition Demo

See [Running the Live Camera Recognition Demo](docs/imagenet-camera.md)

### Re-training the Network with DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md)

#### Downloading Image Recognition Dataset

See [Re-Training the Recognition Network](docs/imagenet-training.md#downloading-image-recognition-dataset)

#### Customizing the Object Classes

See [Re-Training the Recognition Network](docs/imagenet-training.md#customizing-the-object-classes)

#### Importing Classification Dataset into DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#importing-classification-dataset-into-digits)

#### Creating Image Classification Model with DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#creating-image-classification-model-with-digits)

#### Testing Classification Model in DIGITS

See [Re-Training the Recognition Network](docs/imagenet-training.md#testing-classification-model-in-digits)

#### Downloading Model Snapshot to Jetson

See [Downloading Model Snapshots to Jetson](docs/imagenet-snapshot.md)

### Loading Custom Models on Jetson

See [Loading Custom Models on Jetson](docs/imagenet-custom.md)

### Locating Object Coordinates using DetectNet

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md)

#### Detection Data Formatting in DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#detection-data-formatting-in-digits)

#### Downloading the Detection Dataset

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#downloading-the-detection-dataset)

#### Importing the Detection Dataset into DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#importing-the-detection-dataset-into-digits)

#### Creating DetectNet Model with DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#creating-detectnet-model-with-digits)

#### Selecting DetectNet Batch Size

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#selecting-detectnet-batch-size)

#### Specifying the DetectNet Prototxt 

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#specifying-the-detectnet-prototxt)

#### Training the Model with Pretrained Googlenet

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#training-the-model-with-pretrained-googlenet)

#### Testing DetectNet Model Inference in DIGITS

See [Locating Object Coordinates using DetectNet](docs/detectnet-training.md#testing-detectnet-model-inference-in-digits)

#### Downloading the Model Snapshot to Jetson

See [Downloading the Detection Model to Jetson](docs/detectnet-snapshot.md)

#### DetectNet Patches for TensorRT

See [Downloading the Detection Model to Jetson](docs/detectnet-snapshot.md#detectnet-patches-for-tensorrt)

### Processing Images from the Command Line on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md)

#### Launching With a Pretrained Model

See [Detecting Objects from the Command Line](docs/detectnet-console.md#launching-with-a-pretrained-model)

#### Pretrained DetectNet Models Available

See [Detecting Objects from the Command Line](docs/detectnet-console.md#pretrained-detectnet-models-available)

#### Running Other MS-COCO Models on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md#running-other-ms-coco-models-on-jetson)

#### Running Pedestrian Models on Jetson

See [Detecting Objects from the Command Line](docs/detectnet-console.md#running-pedestrian-models-on-jetson)

#### Multi-class Object Detection Models

See [Detecting Objects from the Command Line](docs/detectnet-console.md#multi-class-object-detection-models)

### Running the Live Camera Detection Demo on Jetson

See [Running the Live Camera Detection Demo](docs/detectnet-camera.md)

### Image Segmentation with SegNet

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md)

#### Downloading Aerial Drone Dataset

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md#downloading-aerial-drone-dataset)

#### Importing the Aerial Dataset into DIGITS

See [Semantic Segmentation with SegNet](docs/segnet-dataset.md#importing-the-aerial-dataset-into-digits)

#### Generating Pretrained FCN-Alexnet

See [Generating Pretrained FCN-Alexnet](docs/segnet-pretrained.md)

### Training FCN-Alexnet with DIGITS

See [Training FCN-Alexnet with DIGITS](docs/segnet-training.md)

#### Testing Inference Model in DIGITS

See [Training FCN-Alexnet with DIGITS](docs/segnet-training.md#testing-inference-model-in-digits)

#### FCN-Alexnet Patches for TensorRT

See [FCN-Alexnet Patches for TensorRT](docs/segnet-patches.md)

### Running Segmentation Models on Jetson

See [Running Segmentation Models on Jetson](docs/segnet-console.md)

</details>

