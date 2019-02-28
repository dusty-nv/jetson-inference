<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="building-repo.md">Back</a> | <a href="imagenet-camera.md">Next</a> | </sup><b><a href="../README.md"><sup>Contents</sup></a></b></p>  

## Classifying Images with ImageNet
There are multiple types of deep learning networks available, including recognition, detection/localization, and soon segmentation.  The first deep learning capability we're highlighting in this tutorial is **image recognition** using an 'imageNet' that's been trained to identify similar objects.

The [`imageNet`](imageNet.h) object accepts an input image and outputs the probability for each class.  Having been trained on ImageNet database of **[1000 objects](data/networks/ilsvrc12_synset_words.txt)**, the standard AlexNet and GoogleNet networks are downloaded during [step 2](#configuring-with-cmake) from above.  As examples of using [`imageNet`](imageNet.h) we provide a command-line interface called [`imagenet-console`](imagenet-console/imagenet-console.cpp) and a live camera program called [`imagenet-camera`](imagenet-camera/imagenet-camera.cpp).

### Using the Console Program on Jetson

First, try using the [`imagenet-console`](imagenet-console/imagenet-console.cpp) program to test imageNet recognition on some example images.  It loads an image, uses TensorRT and the [`imageNet`](imageNet.h) class to perform the inference, then overlays the classification and saves the output image.

After [building](#building-from-source-on-jetson), make sure your terminal is located in the aarch64/bin directory:

``` bash
$ cd jetson-inference/build/aarch64/bin
```

Then, classify an example image with the [`imagenet-console`](imagenet-console/imagenet-console.cpp) program.  [`imagenet-console`](imagenet-console/imagenet-console.cpp) accepts 2 command-line arguments:  the path to the input image and path to the output image (with the class overlay printed).

``` bash
$ ./imagenet-console orange_0.jpg output_0.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange.jpg" width="500">

``` bash
$ ./imagenet-console granny_smith_1.jpg output_1.jpg
```
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple.jpg" width="500">

Next, we will use [imageNet](imageNet.h) to classify a live video feed from the Jetson onboard camera.

##
<p align="right">Next | <b><a href="imagenet-camera.md">Running the Live Camera Recognition Demo</a></b>
<br/>
Back | <b><a href="building-repo.md">Building the Repo from Source</a></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><b><a href="../README.md"><sup>Table of Contents</sup></a></b></p>
