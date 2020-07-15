<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="digits-workflow.md">Back</a> | <a href="jetpack-setup.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>System Setup</sup></p>  

# Natively setting up DIGITS on the Host 

> **note**:  it is recommended for beginners to setup DIGITS with [NVIDIA GPU Cloud (NGC)](digits-setup.md) 

If you chose not to use NGC container for DIGITS, you need to natively set up your CUDA development environment on your PC and build DIGITS.

#### Installing NVIDIA Driver on the Host

At this point, JetPack will have flashed the Jetson with the latest L4T BSP, and installed CUDA toolkits to both the Jetson and host PC.  However, the NVIDIA PCIe driver will still need to be installed on the host PC to enable GPU-accelerated training.  Run the following commands from the host PC to install the NVIDIA driver from the Ubuntu repo:

``` bash
$ sudo apt-get install nvidia-384	# use nvidia-375 for alternate version
$ sudo reboot
```

Afer rebooting, the NVIDIA driver should be listed under `lsmod`:

``` bash
$ lsmod | grep nvidia
nvidia_uvm            647168  0
nvidia_drm             49152  1
nvidia_modeset        790528  4 nvidia_drm
nvidia              12144640  60 nvidia_modeset,nvidia_uvm
drm_kms_helper        167936  1 nvidia_drm
drm                   368640  4 nvidia_drm,drm_kms_helper
```

To verify the CUDA toolkit and NVIDIA driver are working, run some tests that come with the CUDA samples:

``` bash
$ cd /usr/local/cuda/samples
$ sudo make
$ cd bin/x86_64/linux/release/
$ ./deviceQuery
$ ./bandwidthTest --memory=pinned
```

#### Installing cuDNN on the Host

The next step is to install NVIDIA **[cuDNN](https://developer.nvidia.com/cudnn)** libraries on the host PC.  Download the libcudnn and libcudnn packages from the NVIDIA cuDNN webpage:

[`https://developer.nvidia.com/cudnn`](https://developer.nvidia.com/cudnn)

Then install the packages with the following commands:

``` bash
$ sudo dpkg -i libcudnn<version>_amd64.deb
$ sudo dpkg -i libcudnn-dev_<version>_amd64.deb
```

#### Installing NVcaffe on the Host

[NVcaffe](https://github.com/nvidia/caffe/tree/caffe-0.15) is the NVIDIA branch of Caffe with optimizations for GPU.  NVcaffe requires cuDNN and is used by DIGITS for training DNNs.  To install it, clone the NVcaffe repo from GitHub, and compile from source, using the caffe-0.15 branch.

> **note**: for this tutorial, NVcaffe is only required on the host (for training).  During inferencing phase TensorRT is used on the Jetson and doesn't require caffe.

First clone the caffe-0.15 branch from https://github.com/NVIDIA/caffe

``` bash
$ git clone -b caffe-0.15 https://github.com/NVIDIA/caffe
```

Build caffe with the [instructions](http://caffe.berkeleyvision.org/installation.html#compilation) from here:

[`http://caffe.berkeleyvision.org/installation.html#compilation`](http://caffe.berkeleyvision.org/installation.html#compilation)

Caffe should now be configured and built.  Now edit your user's ~/.bashrc to include the path to your Caffe tree (replace the paths below to reflect your own):

``` bash
export CAFFE_ROOT=/home/dusty/workspace/caffe
export PYTHONPATH=/home/dusty/workspace/caffe/python:$PYTHONPATH
```

Close and re-open the terminal for the changes to take effect.


#### Installing DIGITS on the Host

NVIDIA **[DIGITS](https://developer.nvidia.com/digits)** is a Python-based web service which interactively trains DNNs and manages datasets.  As highlighed in the DIGITS workflow, it runs on the host PC to create the network model during the training phase.  The trained model is then copied from the host PC to the Jetson for the runtime inference phase with TensorRT.

For automated installation, it's recommended to use DIGITS through [NVIDIA GPU Cloud](https://www.nvidia.com/en-us/gpu-cloud/), which comes with a DIGITS Docker image that can run on a GPU attached to a local PC or cloud instance. Alternatively, to install DIGITS from source, first clone the DIGITS repo from GitHub:

``` bash
$ git clone https://github.com/nvidia/DIGITS
```

Then complete the steps under the **[Building DIGITS](https://github.com/NVIDIA/DIGITS/blob/digits-6.0/docs/BuildDigits.md)** documentation.

[`https://github.com/NVIDIA/DIGITS/blob/digits-6.0/docs/BuildDigits.md`](https://github.com/NVIDIA/DIGITS/blob/digits-6.0/docs/BuildDigits.md)

#### Starting the DIGITS Server

Assuming that your terminal is still in the DIGITS directory, the webserver can be started by running the `digits-devserver` Python script:

``` bash
$ ./digits-devserver 
  ___ ___ ___ ___ _____ ___
 |   \_ _/ __|_ _|_   _/ __|
 | |) | | (_ || |  | | \__ \
 |___/___\___|___| |_| |___/ 5.1-dev

2017-04-17 13:19:02 [INFO ] Loaded 0 jobs.`
```

DIGITS will store user jobs (training datasets and model snapshots) under the `digits/jobs` directory.

To access the interactive DIGITS session, open your web browser and navigate to `0.0.0.0:5000`.

> **note**:  by default the DIGITS server will start on port 5000, but the port can be specified by passing the `--port` argument to the `digits-devserver` script.

##
<p align="right">Next | <b><a href="jetpack-setup.md">Setting up Jetson with JetPack</a></b>
<br/>
Back | <b><a href="digits-workflow.md">DIGITS Workflow</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
