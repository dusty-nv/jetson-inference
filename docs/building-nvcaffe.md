<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">

# Building nvcaffe

A special branch of caffe is used on TX1 which includes support for FP16.<br />
The code is released in NVIDIA's caffe repo in the experimental/fp16 branch, located here:
> https://github.com/nvidia/caffe/tree/experimental/fp16

#### 1. Installing Dependencies

``` bash
$ sudo apt-get update -y
$ sudo apt-get install cmake -y

# General dependencies
$ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev \
libhdf5-serial-dev protobuf-compiler -y
$ sudo apt-get install --no-install-recommends libboost-all-dev -y

# BLAS
$ sudo apt-get install libatlas-base-dev -y

# Remaining Dependencies
$ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev -y

# Python Dependencies
$ sudo apt-get install python-dev python-numpy python-skimage python-protobuf -y
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
$ rm -rf cmake/ CMakeLists.txt
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

###### Setup header/linker paths

``` bash
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
    becomes
    INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/

    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib
    becomes
    LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/aarch64-linux-gnu/hdf5/serial/
```

###### Fix HDF5 Linking Issue

``` bash
$ sudo ln -s /usr/lib/aarch64-linux-gnu/libhdf5_serial.so.10 /usr/lib/aarch64-linux-gnu/libhdf5.so
$ sudo ln -s /usr/lib/aarch64-linux-gnu/libhdf5_serial_hl.so.10 /usr/lib/aarch64-linux-gnu/libhdf5_hl.so
```

#### 4. Compiling nvcaffe

``` bash
$ make -j4
$ make pycaffe
$ make distribute
```

#### 5. Set environment paths

``` bash
# Edit the following lines in .bashrc and then source .bashrc
export PATH=/usr/local/cuda-8.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/ubuntu/caffe/build/tools
export PYTHONPATH=/home/ubuntu/caffe/python:$PYTHONPATH
```

#### 6. Install iPython (optional)

``` bash
$ sudo apt-get install ipython ipython-notebook python-pandas -y
```

#### 6. Testing nvcaffe

``` bash
$ make runtest
```
