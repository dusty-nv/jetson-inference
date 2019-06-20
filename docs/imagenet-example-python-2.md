<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-console-2.md">Back</a> | <a href="imagenet-example-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Coding Your Own Image Recognition Program (Python)
In the previous step, we ran an application that came with the `jetson-inference` repo.  

Now, we're going to walk through creating a new program from scratch in Python for image recognition called [`my-recognition.py`](../python/examples/my-recognition.py).  This script will load an arbitrary image from disk and classify it using the [`imageNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#imageNet) object.  

For your convenience and reference, the completed source is available in the [`python/examples/my-recognition.py`](../python/examples/my-recognition.py) file of the repo, but the guide below will act like they reside in the user's home directory or in an arbitrary directory of your choosing.   

## Setting up the Project

You can store the `my-recognition.py` example that we will be creating wherever you want on your Jetson.  

For simplicity, this guide will create it along with some test images inside a directory under the user's home directory located at `~/my-recognition`.

Run these commands from a terminal to create the directory and files required:  

``` bash
$ cd ~/
$ mkdir my-recognition
$ cd my-recognition
$ touch my-recognition.py
$ chmod +x my-recognition.py
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/black_bear.jpg 
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/brown_bear.jpg
$ wget https://github.com/dusty-nv/jetson-inference/raw/master/data/images/polar_bear.jpg 
```

Some test images are also downloaded to the folder with the `wget` commands above.  

Next, we'll add the Python code for the program to the empty source file we created here.

## Source Code

Open up `my-recognition.py` in your editor of choice (or run `gedit my-recognition.py`).  

First, let's add a shebang sequence to the very top of the file to automatically use the Python interpreter:

``` python
#!/usr/bin/python
```

Next, we'll import the Python modules that we're going to use in the script.

### Importing Modules

Add `import` statements to load the [`jetson.inference`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html) and [`jetson.utils`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.utils.html) modules used for recognizing images and image loading.  We'll also load the standard `argparse` package for parsing the command line.

``` python
import jetson.inference
import jetson.utils

import argparse
```

> **note**:  these Jetson modules are installed during the `sudo make install` step of [building the repo](building-repo-2.md#compiling-the-project).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if you did not run `sudo make install`, then these packages won't be found when we go to run the example.  


### Parsing the Command Line

Next, add some boilerplate code to parse the image filename and an optional `--network` parameter:

``` python
# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()
```

This example loads and classifies an image that the user specifies.  It will be expected to be run like this:

``` bash
$ ./my-recognition.py my_image.jpg
```

The desired image filename to be loaded should be substituted for `my_image.jpg`.  You can also optionally specify the `--network` parameter to change the classification network that's used (the default is GoogleNet):

``` bash
$ ./my-recognition.py --network=resnet-18 my_image.jpg
```

See the [Downloading Other Classification Models](imagenet-console-2.md#downloading-other-classification-models] section from the previous page for more information about downloading other recognition networks.


### Loading the Image from Disk

You can load images from disk into GPU memory using the `loadImageRGBA()` function.  JPG, PNG, TGA, and BMP formats are supported.

Add this line to load the image with the filename that was specified from the command line:

``` python
img, width, height = jetson.utils.loadImageRGBA(opt.filename)
```

The loaded image will be stored in shared memory that's mapped to both the CPU and GPU.  Since the Jetson's CPU and integrated GPU share the same physical memory, memory copies (i.e. `cudaMemcpy()`) between devices aren't needed.  

Note that the image is loaded in `float4` RGBA format, with pixel values between 0.0 and 255.0.  


### Loading the Image Recognition Network

Using the [`imageNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#imageNet) object, the following code will load the desired classification model with TensorRT.  Unless you specified a different network using the `--network` flag, by default it will load GoogleNet, which was already downloaded when you initially [built the `jetson-inference` repo](building-repo-2.md#compiling-the-project) (the `ResNet-18` model was also selected by default to be downloaded).

All of the available classification models are pre-trained on the ImageNet ILSVRC dataset, which can recognize up to [1000 different classes](../data/networks/ilsvrc12_synset_words.txt) of objects, like different kinds of fruits and vegetables, many different species of animals, along with everyday man-made objects like vehicles, office furniture, sporting equipment, ect.   

``` python
# load the recognition network
net = jetson.inference.imageNet(opt.network)
```

#### Classifying the Image

Next, we are going to classify the image with the recognition network using the `imageNet.Classify()` function:  

``` python
# classify the image
class_idx, confidence = net.Classify(img, width, height)
```

`imageNet.Classify()` accepts the image and it's dimensions, and performs the inferencing with TensorRT.  

It returns a tuple containing the integer index of the object class that the image was recognized as, along with the floating-point confidence value of the result.

### Interpreting the Results

As the final step, let's retrieve the class description and print out the results of the classification:

``` python
# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
```

`imageNet.Classify()` returns the index of the recognized object class (between `0` and `999` for these models that were trained on ILSVRC).  Given the class index, the `imageNet.GetClassDesc()` function will then return the string containing the text description of that class.  These descriptions are automatically loaded from [`ilsvrc12_synset_words.txt`](../data/networks/ilsvrc12_synset_words.txt).

That's it!  That is all the code we need.

### Source Listing

For completeness, here is the full source of the Python script that we just created.  You can also find it in the repo at [`python/examples/my-recognition.py`](../python/examples/my-recognition.py).

``` python
#!/usr/bin/python

import jetson.inference
import jetson.utils

import argparse

# parse the command line
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("--network", type=str, default="googlenet", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
opt = parser.parse_args()

# load an image (into shared CPU/GPU memory)
img, width, height = jetson.utils.loadImageRGBA(opt.filename)

# load the recognition network
net = jetson.inference.imageNet(opt.network)

# classify the image
class_idx, confidence = net.Classify(img, width, height)

# find the object description
class_desc = net.GetClassDesc(class_idx)

# print out the result
print("image is recognized as '{:s}' (class #{:d}) with {:f}% confidence".format(class_desc, class_idx, confidence * 100))
```

## Running the Example

Now that our Python program is complete, let's classify the test images that we [downloaded](#setting-up-the-project) at the beginning of this page:  

``` bash
$ ./my-recognition.py polar_bear.jpg
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

You can also use a [different network](imagenet-console-2.md#downloading-other-classification-models) by specifying the `--network` flag, like so:

``` bash
$ ./my-recognition --network=resnet-18 polar_bear.jpg
image is recognized as 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus' (class #296) with 99.743396% confidence
```

Next, we'll walk through the creation of the C++ version of this program.

##
<p align="right">Next | <b><a href="imagenet-example-2.md">Coding Your Own Image Recognition Program (C++)</a></b>
<br/>
Back | <b><a href="imagenet-console-2.md">Classifying Images with ImageNet</a></b></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
