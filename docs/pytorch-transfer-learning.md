<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-camera-2.md">Back</a> | <a href="pytorch-cat-dog.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning</sup></s></p>

# Transfer Learning with PyTorch

Transfer learning is a technique for re-training a DNN model on a new dataset, which takes less time than training a network from scratch.  With transfer learning, the weights of a pre-trained model are fine-tuned to classify a customized dataset.  In these examples, we'll be using the <a href="https://arxiv.org/abs/1512.03385">ResNet-18</a> and [SSD-Mobilenet](pytorch-ssd.md) networks, although you can experiment with other networks too.

<p align="center"><a href="https://arxiv.org/abs/1512.03385"><img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-resnet-18.png" width="600"></a></p>

Although training is typically performed on a PC, server, or cloud instance with discrete GPU(s) due to the often large datasets used and the associated computational demands, by using transfer learning we're able to re-train various networks onboard Jetson to get started with training and deploying our own DNN models.  

<a href=https://pytorch.org/>PyTorch</a> is the machine learning framework that we'll be using, and example datasets along with training scripts are provided to use below, in addition to a camera-based tool for collecting and labelling your own training datasets.  

## Installing PyTorch

If you optionally chose to install PyTorch back when you [built the repo](building-repo-2.md#installing-pytorch), it should already be installed on your Jetson to use.  Otherwise, if you want to proceed with transfer learning, you can install it now:

``` bash
$ cd jetson-inference/build
$ ./install-pytorch.sh
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/pytorch-installer.jpg" width="650">

> **note**: the automated PyTorch installation tool requires JetPack 4.2 or newer.<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;and if you want to do [Object Detection Training](pytorch-ssd.md), you should use JetPack 4.4 or newer and install PyTorch for **Python 3.6**.

### Verifying PyTorch

You can test that PyTorch was installed correctly and detects your GPU by executing these commands from an interactive Python shell - run `python` or `python3` from a terminal:

``` bash
>>> import torch
>>> print(torch.__version__)
>>> print('CUDA available: ' + str(torch.cuda.is_available()))
>>> a = torch.cuda.FloatTensor(2).zero_()
>>> print('Tensor a = ' + str(a))
>>> b = torch.randn(2).cuda()
>>> print('Tensor b = ' + str(b))
>>> c = a + b
>>> print('Tensor c = ' + str(c))
```

``` bash
>>> import torchvision
>>> print(torchvision.__version__)
```

Note that the torch version should be reported as `1.1.0` and the torchvision version should be `0.3.0`.

## Mounting Swap

Unless you are on Jetson AGX Xavier, you should mount 4GB of swap space, as training uses up a lot of extra memory.  

Run these commands on Nano to create a swap file:

``` bash
sudo fallocate -l 4G /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

Then add the following line to the end of `/etc/fstab` to make the change persistent:

``` bash
/mnt/4GB.swap  none  swap  sw 0  0
```

Now your swap file will automatically be mounted after reboots.  To check the usage, run `swapon -s` or `tegrastats`.
 
## Training Datasets

Below are step-by-step instructions to re-training models on some example datasets with transfer learning, in addition to collecting your own data to create your own customized models: 

* Classification/Recognition (ResNet-18)
	* [Re-training on the Cat/Dog Dataset](pytorch-cat-dog.md)
	* [Re-training on the PlantCLEF Dataset](pytorch-plants.md)
	* [Collecting your own Classification Datasets](pytorch-collect.md)
* Object Detection (SSD-Mobilenet)
	* [Re-training SSD-Mobilenet](pytorch-ssd.md)
	* [Collecting your own Detection Datasets](pytorch-collect.md)

This table contains a summary of the datasets and their associated training times:

| Model | Dataset   | Size  |  Classes | Training Images | Time per Epoch* | Training Time** |
|-----------|-----------|-------|----------|-----------------|-----------------|-----------------|
| ResNet-18 | [`Cat/Dog`](pytorch-cat-dog.md)   | 800MB |    2    |      5,000      |  ~7-8 minutes   |    ~4 hours     |
| ResNet-18 | [`PlantCLEF`](pytorch-plants.md) | 1.5GB |   20    |     10,475      | ~15 minutes     |    ~8 hours     |
| SSD-Mobilenet | [`Plants`](pytorch-ssd.md) | 2GB |   8    |     6,375      | ~15 minutes     |    ~8 hours     |

*&nbsp;&nbsp;Approximate time for one complete training pass over the dataset with Jetson Nano  
** Approximate time for training the model for 35 epochs with Jetson Nano

<p align="right">Next | <b><a href="pytorch-cat-dog.md">Re-training on the Cat/Dog Dataset</a></b>
<br/>
Back | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
