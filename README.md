![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/841b9209217f74e5992b8d332c612126)
# Deploying Deep Learning
Welcome to NVIDIA's deep learning inference workshop and end-to-end realtime object recognition library for Jetson TX1.

During this tutorial, we'll deploy efficient neural networks using [GPU Inference Engine](https://developer.nvidia.com/gpu-inference-engine) and stream from live camera feed.

### Table of Contents

* [Table of Contents](#table-of-contents)
* [Introduction](#introduction)
    * [Training](#training)
      * [DIGITS](#digits)
    * [Inference](#inference)
* [Building nvcaffe](#building-nvcaffe)
* [Installing GPU Inference Engine](#installing-gpu-inference-engine)
* [Compiling from Source](#compiling-from-source)
* [Running the Recognition Demo](#running-the-recognition-demo)

> **note**:  this branch of the tutorial is verified against 
>        JetPack 2.2 / L4T R24.1 aarch64.

### Introduction

*Deep-learning* networks typically have two primary phases of development:   **training** and **inference**

#### Training
During the training phase, the network learns from a large dataset of labeled examples.  The weights of the neural network become optimized to recognize the patterns contained within the training dataset.  Deep neural networks have many layers of neurons connected togethers.  Deeper networks take increasingly longer to train and evaluate, but are ultimately able to encode more intelligence within them.

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/fd4ba9e7e68b76fc41c8312856c7d0ad)

Throughout training, the network's inference performance is tested and refined using trial dataset. Like the training dataset, the trial dataset is labeled with ground-truth so the network's accuracy can be evaluated, but was not included in the training dataset.  The network continues to train iteratively until it reaches a certain level of accuracy set by the user.

Due to the size of the datasets and deep inference networks, training is typically very resource-intensive and can take weeks or months on traditional compute architectures.  However, using GPUs vastly accellerates the process down to days or hours.  

##### DIGITS

Using [DIGITS](https://developer.nvidia.com/digits), anyone can easily get started and interactively train their networks with GPU acceleration.  <br />DIGITS is an open-source project contributed by NVIDIA, located here: https://github.com/NVIDIA/DIGITS. 

This tutorial will use DIGITS and Jetson TX1 together for training and deploying deep-learning networks, <br />refered to as the DIGITS workflow:

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/90bde1f85a952157b914f75a9f8739c2)


#### Inference
Using it's trained weights, the network evaluates live data at runtime.  Called inference, the network predicts and applies reasoning based off the examples it learned.  Due to the depth of deep learning networks, inference requires significant compute resources to process in realtime on imagery and other sensor data.  However, using NVIDIA's GPU Inference Engine which uses Jetson's integrated NVIDIA GPU, inference can be deployed onboard embedded platforms.  Applications in robotics like picking, autonomous navigation, agriculture, and industrial inspection have many uses for deploying deep inference, including:

  - Image recognition
  - Object detection
  - Segmentation 
  - Image registration (homography estimation)
  - Depth from raw stereo
  - Signal analytics
  - Others?


## Building nvcaffe

A special branch of caffe is used on TX1 which includes support for FP16.<br />
The code is released in NVIDIA's caffe repo in the experimental/fp16 branch, located here:
> https://github.com/nvidia/caffe/tree/experimental/fp16

#### 1. Installing Dependencies

``` bash
$ sudo apt-get install protobuf-compiler libprotobuf-dev cmake git libboost-thread1.55-dev libgflags-dev libgoogle-glog-dev libhdf5-dev libatlas-dev libatlas-base-dev libatlas3-base liblmdb-dev libleveldb-dev
```

The Snappy package needs a symbolic link created for Caffe to link correctly:

``` bash
$ sudo ln -s /usr/lib/libsnappy.so.1 /usr/lib/libsnappy.so
$ sudo ldconfig
```

#### 2. Clone nvcaffe fp16 branch

``` bash
$ git clone -b experimental/fp16 https://github.com/NVIDIA/caffe
```

This will checkout the repo to a local directory called `caffe` on your Jetson.

#### 3. Setup build options

``` bash
$ cd caffe
$ cp Makefile.config.example Makefile.config
```

###### Enable FP16:

``` bash
$ sed -i 's/# NATIVE_FP16/NATIVE_FP16/g' Makefile.config
```

###### Enable cuDNN:

``` bash
$ sed -i 's/# USE_CUDNN/USE_CUDNN/g' Makefile.config
```

###### Enable compute_53/sm_53:

``` bash 
$ sed -i 's/-gencode arch=compute_50,code=compute_50/-gencode arch=compute_53,code=sm_53 -gencode arch=compute_53,code=compute_53/g' Makefile.config
```

#### 4. Compiling nvcaffe

``` bash
$ make all
$ make test
```

#### 5. Testing nvcaffe

``` bash
$ make runtest
```

## Installing GPU Inference Engine

NVIDIA's [GPU Inference Engine](https://developer.nvidia.com/gie) (GIE) is an optimized backend for evaluating deep inference networks in prototxt format.

#### 1. Package contents

First, unzip the archive:
```
$ tar -zxvf gie.aarch64-cuda7.0-1.0-ea.tar.gz
```

The directory structure is as follows:
```
|-GIE
|  \bin  where the samples are built to
|  \data sample network model / prototxt's
|  \doc  API documentation and User Guide
|  \include
|  \lib 
|  \samples 
```

#### 2. Remove packaged cuDNN

If you flashed your Jetson TX1 with JetPack or already have cuDNN installed, remove the version of cuDNN that comes with GIE:

```
$ cd GIE/lib
$ rm libcudnn*
$ cd ../../
```

#### 3. Build samples

````
$ cd GIE/samples/sampleMNIST
$ make TARGET=tx1
Compiling: sampleMNIST.cpp
Linking: ../../bin/sample_mnist_debug
Compiling: sampleMNIST.cpp
Linking: ../../bin/sample_mnist
$ cd ../sampleGoogleNet
$ make TARGET=tx1
Compiling: sampleGoogleNet.cpp
Linking: ../../bin/sample_googlenet_debug
Compiling: sampleGoogleNet.cpp
Linking: ../../bin/sample_googlenet
$ cd ../../../
````

#### 4. Running samples

````
$ cd GIE/bin
$ ./sample_mnist
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@%+-:  =@@@@@@@@@@@@
@@@@@@@%=      -@@@**@@@@@@@
@@@@@@@   :%#@-#@@@. #@@@@@@
@@@@@@*  +@@@@:*@@@  *@@@@@@
@@@@@@#  +@@@@ @@@%  @@@@@@@
@@@@@@@.  :%@@.@@@. *@@@@@@@
@@@@@@@@-   =@@@@. -@@@@@@@@
@@@@@@@@@%:   +@- :@@@@@@@@@
@@@@@@@@@@@%.  : -@@@@@@@@@@
@@@@@@@@@@@@@+   #@@@@@@@@@@
@@@@@@@@@@@@@@+  :@@@@@@@@@@
@@@@@@@@@@@@@@+   *@@@@@@@@@
@@@@@@@@@@@@@@: =  @@@@@@@@@
@@@@@@@@@@@@@@ :@  @@@@@@@@@
@@@@@@@@@@@@@@ -@  @@@@@@@@@
@@@@@@@@@@@@@# +@  @@@@@@@@@
@@@@@@@@@@@@@* ++  @@@@@@@@@
@@@@@@@@@@@@@*    *@@@@@@@@@
@@@@@@@@@@@@@#   =@@@@@@@@@@
@@@@@@@@@@@@@@. +@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@

0:
1:
2:
3:
4:
5:
6:
7:
8: **********
9:
````
The MNIST sample randomly selects an image of a numeral 0-9, which is then classified with the MNIST network using GIE.  In this example, the network correctly recognized the image as #8.


## Compiling from Source
Provided along with this tutorial are examples of running Googlenet/Alexnet on live camera feed, for object recognition.

#### 1. Cloning the repo
To obtain the repository, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed locally:

``` bash
sudo apt-get install git cmake
```

Then clone the jetson-inference repo:
``` bash
git clone http://github.org/dusty-nv/jetson-inference
```

#### 2. Configuring build

When cmake is run, a special pre-installation script (CMakePreBuild.sh) is run and will automatically install any dependencies.

``` bash
mkdir build
cd build
cmake ../
```

#### 3. Compiling

Make sure you are still in the jetson-inference/build directory, created above in step #2.

``` bash
cd jetson-inference/build			# omit if pwd is already /build from above
make
```

Depending on architecture, the package will be built to either armhf or aarch64, with the following directory structure:

```
|-build
   \aarch64		    (64-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```

binaries residing in aarch64/bin, headers in aarch64/include, and libraries in aarch64/lib.

## Running the Recognition Demo

The realtime image recognition demo is located in /aarch64/bin and is called imagenet-camera.
It runs on live camera stream and depending on user arguments, loads googlenet or alexnet with GPU Inference Engine: 
``` bash
$ cd jetson-inference/build/aarch64/bin

$ ./imagenet-camera googlenet           # to run using googlenet
$ ./imagenet-camera alexnet             # to run using alexnet
```

The frames per second (FPS), classified object name from the video, and confidence of the classified object are printed to the openGL window title bar.  By default the application can recognize up to 1000 different types of objects, since Googlenet and Alexnet are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find included in the repo under [data/networks/ilsvrc12_synset_words.txt](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd" width="800"></a>
<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3" width="800"></a>

