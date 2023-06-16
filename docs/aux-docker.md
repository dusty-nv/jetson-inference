<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="jetpack-setup-2.md">Back</a> | <a href="building-repo-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p>  

# Running the Docker Container

Pre-built Docker container images for this project are hosted on [DockerHub](https://hub.docker.com/r/dustynv/jetson-inference/tags).  Alternatively, you can [Build the Project ](building-repo-2.md) from source.   

Below are the currently available container tags:

| Container Tag                                                                           | L4T version |          JetPack version         |
|-----------------------------------------------------------------------------------------|:-----------:|:--------------------------------:|
| [`dustynv/jetson-inference:r35.3.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R35.3.1 | JetPack 5.1.1 |
| [`dustynv/jetson-inference:r35.2.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R35.2.1 | JetPack 5.1 |
| [`dustynv/jetson-inference:r35.1.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R35.1.0 | JetPack 5.0.2 |
| [`dustynv/jetson-inference:r34.1.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R34.1.1 | JetPack 5.0.1 |
| [`dustynv/jetson-inference:r32.7.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.7.1 | JetPack 4.6.1 |
| [`dustynv/jetson-inference:r32.6.1`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.6.1 | JetPack 4.6 |
| [`dustynv/jetson-inference:r32.5.0`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.5.0 | JetPack 4.5 |
| [`dustynv/jetson-inference:r32.4.4`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.4.4 | JetPack 4.4.1 |
| [`dustynv/jetson-inference:r32.4.3`](https://hub.docker.com/r/dustynv/jetson-inference/tags) | L4T R32.4.3 | JetPack 4.4 |


> **note:** the version of JetPack-L4T that you have installed on your Jetson needs to be compatible with one of the tags above.  If you have a different version of JetPack installed, either upgrade to the latest JetPack or [Build the Project from Source](docs/building-repo-2.md) to compile the project directly. 

These containers use the [`l4t-pytorch`](https://ngc.nvidia.com/catalog/containers/nvidia:l4t-pytorch) base container, so support for training models and transfer learning is already included.

## Launching the Container

Due to various mounts and devices needed to run the container, it's recommended to use the [`docker/run.sh`](../docker/run.sh) script to run the container:

```bash
$ git clone --recursive --depth=1 https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ docker/run.sh
```

> **note:**  because of the Docker scripts used and the data directory structure that gets mounted into the container, you should still clone the project on your host device (i.e. even if not intending to build/install the project natively)

[`docker/run.sh`](../docker/run.sh) will automatically pull the correct container tag from DockerHub based on your currently-installed version of JetPack-L4T, and mount the appropriate data directories and devices so that you can use cameras/display/ect from within the container.

### ROS Support

This project also has the [ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning) package available for ROS/ROS2, and by specifying the `--ros=ROS_DISTRO` option you can start the version of container built with ROS.  Supported ROS distros include Noetic, Foxy, Galactic, Humble, and Iron:

``` bash
$ docker/run.sh --ros=humble   # noetic, foxy, galactic, humble, iron
```

The container will source the ROS environment and packages when started.  For more information, see the [ros_deep_learning](https://github.com/dusty-nv/ros_deep_learning) documentation.

### x86 Support

In addition to being supported on the Jetson ARM-based architectures, the jetson-inference container can be [built](#building-the-container) and run on x86_64 systems with NVIDIA GPU(s).  This can be used to run the Hello AI World tutorial and accompanying apps/libraries from it, or for faster training on the PC/server.  To do this, first install the [NVIDIA drivers](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#pre-requisites) and [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/nvidia-docker.html) to enable GPU support in Docker.  

To run the latest pre-built jetson-inference x86 container, use the same commands as above (`docker/run.sh`).  If you want to use a newer/older version of the [`nvcr.io/nvidia/pytorch`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) base container, edit [this line](https://github.com/dusty-nv/jetson-inference/blob/master/docker/tag.sh#L40) with the desired tag and then run [`docker/build.sh`](#building-the-container)

Although the jetson-inference container is built for Linux, it can be run on Windows under WSL 2 by following the [CUDA on WSL User Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#ch02-sub03-installing-wsl2), followed by installing Docker and the NVIDIA Container Runtime as above.  If you need to use USB webcams and V4L2 under WSL 2, you'll also need to recompile your WSL kernel with these [config changes](https://github.com/PINTO0309/wsl2_linux_kernel_usbcam_enable_conf).

### Mounted Data Volumes

For reference, the following paths automatically get mounted from your host device into the container:

* `jetson-inference/data` (stores the network models, serialized TensorRT engines, and test images)
* `jetson-inference/python/training/classification/data` (stores classification training datasets)
* `jetson-inference/python/training/classification/models` (stores classification models trained by PyTorch)
* `jetson-inference/python/training/detection/ssd/data` (stores detection training datasets)
* `jetson-inference/python/training/detection/ssd/models` (stores detection models trained by PyTorch)

These mounted volumes assure that the models and datasets are stored outside the container, and aren't lost when the container is shut down.

If you wish to mount your own directory into the container, you can use the `--volume HOST_DIR:MOUNT_DIR` argument to [`docker/run.sh`](../docker/run.sh):

```bash
$ docker/run.sh --volume /my/host/path:/my/container/path    # these should be absolute paths
```

You can specify `--volume` multiple times to mount multiple directories.  For more info, run or see [`docker/run.sh --help`](../docker/run.sh)

## Running Applications

Once the container is up and running, you can then run example programs from the tutorial like normal inside the container:

```bash
$ cd build/aarch64/bin
$ ./video-viewer /dev/video0
$ ./imagenet images/jellyfish.jpg images/test/jellyfish.jpg
$ ./detectnet images/peds_0.jpg images/test/peds_0.jpg
# (press Ctrl+D to exit the container)
```

> **note:** when you are saving images from one of the sample programs (like imagenet or detectnet), it's recommended to save them to `images/test`.  These images will then be easily viewable from your host device in the `jetson-inference/data/images/test` directory.  

## Building the Container

If you are following the Hello AI World tutorial, you can ignore this section and skip ahead to the next step.  But if you wish to re-build the container or build your own, you can use the [`docker/build.sh`](../docker/build.sh) script which builds the project's [`Dockerfile`](../Dockerfile):

```bash
$ docker/build.sh
```

>  **note:** you should first set your default `docker-runtime` to nvidia, see [here](https://github.com/dusty-nv/jetson-containers#docker-default-runtime) for the details.

You can also base your own container on this one by using the line `FROM dustynv/jetson-inference:rXX.X.X` in your own Dockerfile.

## Getting Started

If you have chosen to run the project inside the Docker container, you can proceed to [Classifying Images with ImageNet](imagenet-console-2.md).

However, if you would prefer to install the project directly on your Jetson (outside of container), go to [Building the Project from Source](building-repo-2.md).
 
##
<p align="right">Next | <b><a href="building-repo-2.md">Building the Project from Source</a></b>
<br/>
Back | <b><a href="jetpack-setup-2.md">Setting up Jetson with JetPack</a></p>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

