![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/841b9209217f74e5992b8d332c612126)
# Building nvcaffe

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
