<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Other Examples

This area lists other deep-learning resources on inference that are available for Jetson TX1<br />

## GPU Inference Engine (GIE) samples
### Installing GPU Inference Engine

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
