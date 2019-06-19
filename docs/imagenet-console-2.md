<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="building-repo-2.md">Back</a> | <a href="imagenet-example-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p>  

# Classifying Images with ImageNet
There are multiple types of deep learning networks available, including recognition, detection/localization, and semantic segmentation.  The first deep learning capability we're highlighting in this tutorial is **image recognition** using image classifcation networks that have been trained to identify scenes and objects.

[`imageNet`](../imageNet.h) accepts an input image and outputs the probability for each class.  Having been trained on the ImageNet ILSVRC dataset of **[1000 objects](../data/networks/ilsvrc12_synset_words.txt)**, the GoogleNet and ResNet-18 models were automatically downloaded during the build step.  See below for other classification models that can be downloaded and used as well.

As examples of using [`imageNet`](../imageNet.h) we provide versions of a command-line interface for C++ and Python called [`imagenet-console.cpp`](../examples/imagenet-console/imagenet-console.cpp) (C++) and [`imagenet-console.py`](../python/examples/imagenet-console.py) (Python).  There's also versions of a live camera recognition program for C++ and Python called [`imagenet-camera.cpp`](../examples/imagenet-camera/imagenet-camera.cpp) and [`imagenet-camera.py`](../python/examples/imagenet-camera.py).

### Using the Console Program on Jetson

First, try using the [`imagenet-console`](../examples/imagenet-console/imagenet-console.cpp) program to test imageNet recognition on some example images.  It loads an image, uses TensorRT and the [`imageNet`](../imageNet.h) class to perform the inference, then overlays the classification result and saves the output image.

After [building](#building-repo-2.md), make sure your terminal is located in the aarch64/bin directory:

``` bash
$ cd jetson-inference/build/aarch64/bin
```

Then, classify an example image with the [`imagenet-console`](../examples/imagenet-console/imagenet-console.cpp) program.  [`imagenet-console`](../examples/imagenet-console/imagenet-console.cpp) accepts 3 command-line arguments:  the path to the input image and path to the output image, along with an optional `--network` flag which changes the classificaton model being used (the default network is GoogleNet).

#### C++
``` bash
$ ./imagenet-console --network=googlenet orange_0.jpg output_0.jpg  # --network flag is optional
```

#### Python
``` bash
$ ./imagenet-console.py --network=googlenet orange_0.jpg output_0.jpg  # --network flag is optional
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange.jpg" width="500">

> **note**:  the first time you run the program, TensorRT may take up to a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is cached to disk after the first run, so future runs will load faster.

#### C++
``` bash
$ ./imagenet-console granny_smith_1.jpg output_1.jpg
```

#### Python
``` bash
$ ./imagenet-console.py granny_smith_1.jpg output_1.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple.jpg" width="500">



By default, the repo is set to download the GoogleNet and ResNet-18 networks during the build step.

There are others that you can use as well, if you choose to download them:

| Network       | CLI argument   | NetworkType enum |
| --------------|----------------|------------------|
| AlexNet       | `alexnet`      | `ALEXNET`        |
| GoogleNet     | `googlenet`    | `GOOGLENET`      |
| GoogleNet-12  | `googlenet-12` | `GOOGLENET_12`   |
| ResNet-18     | `resnet-18`    | `RESNET_18`      |
| ResNet-50     | `resnet-50`    | `RESNET_50`      |
| ResNet-101    | `resnet-101`   | `RESNET_101`     |
| ResNet-152    | `resnet-152`   | `RESNET_152`     |
| VGG-16        | `vgg-16`       | `VGG-16`         |
| VGG-19        | `vgg-19`       | `VGG-19`         |
| Inception-v4  | `inception-v4` | `INCEPTION_V4`   |

> **note**:  to download additional networks, run the [`download-models.sh`](../tools/download-models.sh) script:  <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>

Generally the more complex networks can have greater accuracy, with increased runtime.

### Using Different Classification Models

You can specify which model to use at runtime by setting the `--network` flag on the command line.

Below are some examples of using the ResNet-18 model:

#### C++
``` bash
$ ./imagenet-console --network=resnet-18 jellyfish.jpg output_jellyfish.jpg
```

#### Python
``` bash
$ ./imagenet-console.py --network=resnet-18 jellyfish.jpg output_jellyfish.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_jellyfish.jpg" width="650">

#### C++
``` bash
$ ./imagenet-console --network=resnet-18 stingray.jpg output_stingray.jpg
```

#### Python
``` bash
$ ./imagenet-console.py --network=resnet-18 stingray.jpg output_stingray.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_stingray.jpg" width="650">

#### C++
``` bash
$ ./imagenet-console.py --network=resnet-18 coral.jpg output_coral.jpg
```

#### Python
``` bash
$ ./imagenet-console.py --network=resnet-18 coral.jpg output_coral.jpg
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/imagenet_coral.jpg" width="650">


Next, we'll go through the steps to code your own image recognition program from scratch, first in Python and then C++.

##
<p align="right">Next | <b><a href="imagenet-example-2.md">Coding Your Own Image Recognition Program</a></b>
<br/>
Back | <b><a href="building-repo-2.md">Building the Repo from Source</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
