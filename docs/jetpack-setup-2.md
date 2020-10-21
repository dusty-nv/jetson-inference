<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="../README.md#hello-ai-world">Back</a> | <a href="aux-docker.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p> 

# Setting up Jetson with JetPack

> **note**:  if your Jetson Nano or Xavier NX has already been setup with the [SD card image](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write) (which includes JetPack), or your Jetson has already been setup with JetPack, you can skip this step and continue to [`Running the Docker Container`](aux-docker.md) or [`Building the Project`](building-repo-2.md)

NVIDIA **[JetPack](https://developer.nvidia.com/embedded/jetpack)** is a comprehensive SDK for Jetson for both developing and deploying AI and computer vision applications.  JetPack simplifies installation of the OS and drivers and contains the following components:

- L4T Kernel / BSP
- CUDA Toolkit
- cuDNN
- TensorRT
- OpenCV
- VisionWorks
- Multimedia API's

Before attempting to use the Docker container or build the repo, make sure that your Jetson has been setup with the latest version of JetPack.

### Jetson Nano and Jetson Xavier NX

The recommended install method for the Jetson Nano Developer Kit and Jetson Xavier NX Developer Kit is to use the **[SD card images](https://developer.nvidia.com/embedded/downloads)**.  

It comes pre-populated with the JetPack components already installed and can be flashed from a Windows, Mac, or Linux PC.  If you haven't already, follow the Getting Started guide for your respective Jetson to flash the SD card image and setup your device:

* [Getting Started with Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit)
* [Getting Started with Jetson Nano 2GB Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-2gb-devkit)
* [Jetson Xavier NX User Guide](https://developer.nvidia.com/embedded/downloads#?search=Jetson%20Xavier%20NX%20Developer%20Kit%20User%20Guide) 

### Jetson TX1/TX2 and AGX Xavier

Other Jetson's should be flashed by downloading the [NVIDIA SDK Manager](https://developer.nvidia.com/embedded/dlc/nv-sdk-manager) to a host PC running Ubuntu 16.04 x86_64 or Ubuntu 18.04 x86_64.  Connect the Micro-USB or USB-C port to your host PC and enter the device into Recovery Mode.

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/nvsdkm.png" width="800">

For more details, please refer to the **[NVIDIA SDK Manager Documentation](https://docs.nvidia.com/sdk-manager/index.html)**.

### Getting the Project

There are two ways to use the jetson-inference project:   

* Run the pre-built [Docker Container](aux-docker.md)
* [Build the Project from Source](building-repo-2.md)

Using the container is recommended initially to get up & running as fast as possible (and the container already includes PyTorch installed), however if you are more comfortable with native development then compiling the project yourself is not complicated either.

##
<p align="right">Next | <b><a href="building-repo-2.md">Building the Project from Source</a></b>
<br/>
Back | <b><a href="../README.md#hello-ai-world">Overview</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
