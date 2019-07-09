<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="detectnet-camera-2.md">Back</a> | <a href="pytorch-cat-dog.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning</sup></s></p>

# Transfer Learning with PyTorch

Transfer learning is a technique for re-training a DNN model on a new dataset, which takes less time than training a network from scratch.  With transfer learning, the weights of a pre-trained model are fine-tuned to classify a customized dataset.  In these examples, we'll be using the <a href="https://arxiv.org/abs/1512.03385">ResNet-18</a> network, although you can experiment with other networks too.

<p align="center"><a href="https://arxiv.org/abs/1512.03385"><img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-resnet-18.png" width="600"></a></p>

Although training is typically performed on a PC, server, or cloud instance with discrete GPU(s) due to the often large datasets used and the associated computational demands, by using transfer learning we're able to re-train various networks onboard Jetson to get started with training our own models.  

<a href=https://pytorch.org/>PyTorch</a> is the machine learning framework that we'll be using, and example datasets along with training scripts are provided to use below, in addition to a camera-based tool for collecting and labelling your own data captured from a live camera feed.  

## Installing PyTorch

If you optionally chose to install PyTorch when you [built the repo](building-repo-2.md#installing-pytorch), then it should already be installed on your Jetson for you.  Otherwise, if you want to proceed with transfer learning, you can install it now:

``` bash
$ cd jetson-inference/build
$ ./install-pytorch.sh
```

<img src="https://raw.githubusercontent.com/dusty-nv/jetson-inference/python/docs/images/download-models.jpg" width="650">

> **note**: the automated PyTorch installation tool requires JetPack 4.2 or newer<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;for other versions, see [`http://eLinux.org/Jetson_Zoo`](https://elinux.org/Jetson_Zoo#PyTorch_.28Caffe2.29) to build from source.

### Verifying PyTorch

You can test that PyTorch was installed correctly and detects your Jetson's GPU by running these commands from a Python interactive shell:

``` bash
$ python	 # substitute 'python3' for Python 3.6
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

The torch version should be reported as `1.1.0` and the torchvision version should be `0.3.0`.

## Mounting Swap

If you are using Jetson Nano, you should mount 4GB of swap space as training uses up a lot of extra memory.  Run these commands on Nano to create a swap file:

``` bash
sudo fallocate -l 4G /mnt/4GB.swap
sudo mkswap /mnt/4GB.swap
sudo swapon /mnt/4GB.swap
```

Then add the following line to the end of `/etc/fstab` to make the change persistent:

``` bash
/mnt/4GB.swap  none  swap  sw 0  0
```

Now your swap file will automatically be mounted after reboots.  To check the usage, run `swapon -s` or `sudo tegrastats`.
 
## Training Datasets

Below are step-by-step instructions to re-training models on some example datasets with transfer learning, in addition to collecting your own data to create your own customized models: 

* [Re-training on Cat/Dog Dataset](pytorch-cat-dog.md)
* [Re-training on PlantCLEF Dataset](pytorch-plants.md)
* [Collecting your own Datasets](pytorch-collect.md)

<p align="right">Next | <b><a href="pytorch-cat-dog.md">Re-training on the Cat/Dog Dataset</a></b>
<br/>
Back | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
