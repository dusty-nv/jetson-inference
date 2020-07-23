<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-console.md">Back</a> | <a href="imagenet-camera.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Coding Your Own Image Recognition Program
In the previous step, we ran an application that came with the jetson-inference repo.  

Now, we're going to walk through creating a new program from scratch for image recognition called [`my-recognition`](../examples/my-recognition/my-recognition.cpp).  This program will be able to exist as a standalone project outside the repo, hence if you wish to use the jetson-inference library in your own projects and applications, you can follow this example.  

For your convenience and reference, the completed files are available in the [`examples/my-recognition`](../examples/my-recognition) directory of the repo, but the guide below will act like they reside in the user's home directory or in an arbitrary directory of your choosing.   

## Setting up the Project

You can store the `my-recognition` example that we will be creating wherever you want on your Jetson.  

For simplicity, this guide will create it in the user's home directory located at `~/my-recognition`.

Run these commands from a terminal to create the directory and files required:  

``` bash
$ mkdir ~/my-recognition
$ cd ~/my-recognition
$ touch my-recognition.cpp
$ touch CMakeLists.txt
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/black_bear.jpg 
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/brown_bear.jpg
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg 
```

Some test images are also downloaded to the folder with the `wget` commands above.  

Next, we'll add the code for the program to the empty source files we created here.

## Source Code

Open up `my-recognition.cpp` in your editor of choice (or run `gedit my-recognition.cpp`).  

Let's start adding the necessary code to use the [`imageNet`](../c/imageNet.h) class for recognizing images.

#### Includes

First, include a couple of headers that we'll need:

``` cpp
// include imageNet header for image recognition
#include <jetson-inference/imageNet.h>

// include loadImage header for loading images
#include <jetson-utils/loadImage.h>
```
> **note**:  these headers are installed under `/usr/local/include` during the `sudo make install` step of [building the repo](building-repo.md#compiling-the-project)  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if you did not run `sudo make install`, then these headers won't be found when we go to compile the example.  

#### Declaring main() and Parsing the Command Line

Next, declare your `main()` method and verify that the program was launched with the image filename as an argument on the command line:  

``` cpp
// main entry point
int main( int argc, char** argv )
{
	// a command line argument containing the image filename is expected,
	// so make sure we have at least 2 args (the first arg is the program)
	if( argc < 2 )
	{
		printf("my-recognition:  expected image filename as argument\n");
		printf("example usage:   ./my-recognition my_image.jpg\n");
		return 0;
	}

	// retrieve the image filename from the array of command line args
	const char* imgFilename = argv[1];
```

This example loads and classifies an image that the user specifies.  It will be expected to be run like this:

``` bash
$ ./my-recognition my_image.jpg
```

The desired image filename to be loaded should be substituted for `my_image.jpg`.  The code above makes sure that this command line argument was provided to the program.

#### Loading the Image from Disk

Declare some variables that will store the dimensions of the image and pointers to it's memory, and then load the image from disk with the [`loadImageRGBA()`](https://github.com/dusty-nv/jetson-utils/blob/master/loadImage.h#L30) function.

``` cpp
	// these variables will be used to store the image data and dimensions
	// the image data will be stored in shared CPU/GPU memory, so there are
	// pointers for the CPU and GPU (both reference the same physical memory)
	float* imgCPU    = NULL;    // CPU pointer to floating-point RGBA image data
	float* imgCUDA   = NULL;    // GPU pointer to floating-point RGBA image data
	int    imgWidth  = 0;       // width of the image (in pixels)
	int    imgHeight = 0;       // height of the image (in pixels)
		
	// load the image from disk as float4 RGBA (32 bits per channel, 128 bits per pixel)
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}
```

The loaded image will be stored in shared memory that's mapped to both the CPU and GPU.  There are two pointers available for access in the CPU and GPU address spaces, but there is really only one copy of the image in memory.  Both the CPU and GPU pointers resolve to the same physical memory, without needing to perform memory copies (i.e. `cudaMemcpy()`).  

When accessing the image from CPU code, the `imgCPU` pointer should be used, and when accessing the image from within a CUDA kernel on the GPU, the `imgCUDA` pointer should be used.  Since our operations in this example will run on the GPU with TensorRT, we will be using the `imgCUDA` pointer.  

The image is loaded in `float4` RGBA format, with pixel values between 0.0 and 255.0.  

#### Loading the Image Recognition Network

Using the [`imageNet::Create()`](../c/imageNet.h#L70) function, the following code will load the GoogleNet model with TensorRT, which was already downloaded when you initially [built the jetson-inference repo](building-repo.md#compiling-the-project).  The model is pre-trained on the ImageNet ILSVRC12 dataset, which can recognize up to [1000 different classes](../data/networks/ilsvrc12_synset_words.txt) of objects, like different kinds of fruits and vegetables, many different species of animals, along with everyday man-made objects like vehicles, office furniture, sporting equipment, ect.   

``` cpp
	// load the GoogleNet image recognition network with TensorRT
	// you can use imageNet::ALEXNET to load AlexNet model instead
	imageNet* net = imageNet::Create(imageNet::GOOGLENET);

	// check to make sure that the network model loaded properly
	if( !net )
	{
		printf("failed to load image recognition network\n");
		return 0;
	}
```

If desired, you can load the AlexNet model instead of GoogleNet by calling [`imageNet::Create(imageNet::ALEXNET)`](../c/imageNet.h#L70).  

The AlexNet model is also trained on the same 1000 classes of objects from ILSVRC12.  

#### Classifying the Image

Next, we are going to classify the image with the image recognition network using the [`imageNet::Classify()`](../c/imageNet.h#L103) function:  

``` cpp
	// this variable will store the confidence of the classification (between 0 and 1)
	float confidence = 0.0;

	// classify the image with TensorRT on the GPU (hence we use the CUDA pointer)
	// this will return the index of the object class that the image was recognized as (or -1 on error)
	const int classIndex = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
```

[`imageNet::Classify()`](../c/imageNet.h#L103) accepts an image pointer in GPU memory, and performs the inferencing with TensorRT.  

It returns the index of the object class that the image was recognized as, along with the confidence value of the result.

#### Interpreting the Results

Unless the call to [`imageNet::Classify()`](../c/imageNet.h#L103) resulted in an error, let's print out the classification info of the recognized object:   

``` cpp
	// make sure a valid classification result was returned	
	if( classIndex >= 0 )
	{
		// retrieve the name/description of the object class index
		const char* classDescription = net->GetClassDesc(classIndex);

		// print out the classification results
		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n", 
			  classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		// if Classify() returned < 0, an error occurred
		printf("failed to classify image\n");
	}
```

Since [`imageNet::Classify()`](../c/imageNet.h#L103) returns an integer-based index of the object class (between 0 and 1000 for ILSVRC12), we use the [`imageNet::GetClassDesc()`](../c/imageNet.h#L140) function to retrieve a human-readable description of the object.  

These descriptions of the 1000 classes are parsed from [`ilsvrc12_synset_words.txt`](../data/networks/ilsvrc12_synset_words.txt) when the network gets loaded (this file was previously downloaded when the jetson-inference repo was built).  

#### Shutting Down

Before exiting the program, `delete` the network object to destroy the TensorRT engine and free CUDA resources:  

``` cpp
	// free the network's resources before shutting down
	delete net;

	// this is the end of the example!
	return 0;
}
```

That's it!  Remember to add the return statement and closing curly bracket to your main() method.  

Next we just need to create a simple makefile for our new recognition program with CMake.

## Creating CMakeLists.txt

Open the file `~/my-recognition/CMakeLists.txt` in editor, and add the following code:

``` cmake
# require CMake 2.8 or greater
cmake_minimum_required(VERSION 2.8)

# declare my-recognition project
project(my-recognition)

# import jetson-inference and jetson-utils packages.
# note that if you didn't do "sudo make install"
# while building jetson-inference, this will error.
find_package(jetson-utils)
find_package(jetson-inference)

# CUDA and Qt4 are required
find_package(CUDA)
find_package(Qt4)

# setup Qt4 for build
include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})

# compile the my-recognition program
cuda_add_executable(my-recognition my-recognition.cpp)

# link my-recognition to jetson-inference library
target_link_libraries(my-recognition jetson-inference)
```

In the future you can use this CMakeLists as a template for compiling your own projects that use the `jetson-inference` library.  The most relevant bits are:

*  Pull in the `jetson-utils` and `jetson-inference` projects:  
     ``` cmake
		find_package(jetson-utils)
		find_package(jetson-inference)
	```
*  Link against `libjetson-inference`:  
     ``` cmake
		target_link_libraries(my-recognition jetson-inference)
	```

> **note**:  these libraries are installed under `/usr/local/lib` during the `sudo make install` step of [building the repo](building-repo.md#compiling-the-project)  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if you did not run `sudo make install`, then these libraries won't be found when we go to compile the example.  

## Building the Example

Now that our source files are complete, run the following shell commands to compile the `my-recognition` program:  

``` bash
$ cd ~/my-recognition
$ cmake .
$ make
```

If you encounter errors, make sure that you ran `sudo make install` while [building the jetson-inference repo](building-repo.md#compiling-the-project).  

You can also download the completed, working code of this example from the [`examples/my-recognition`](../examples/my-recognition) directory of the repo.  

## Running the Example

Now that our program is compiled, let's classify the test images that we [downloaded](#setting-up-the-project) at the beginning of this guide:  

``` bash
$ ./my-recognition polar_bear.jpg
image is recognized as 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus' (class #296) with 99.999878% confidence
```
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg" width="400">

``` bash
$ ./my-recognition brown_bear.jpg
image is recognized as 'brown bear, bruin, Ursus arctos' (class #294) with 99.928925% confidence
```
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/data/images/brown_bear.jpg" width="400">

``` bash
$ ./my-recognition black_bear.jpg
image is recognized as 'American black bear, black bear, Ursus americanus, Euarctos americanus' (class #295) with 98.898628% confidence
```
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/data/images/black_bear.jpg" width="400">

This is the conclusion of this section of the tutorial.  Next, we'll classify a live video feed from the Jetson onboard camera.

##
<p align="right">Next | <b><a href="imagenet-camera.md">Running the Live Camera Recognition Demo</a></b>
<br/>
Back | <b><a href="imagenet-console.md">Classifying Images with ImageNet</a></b></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
