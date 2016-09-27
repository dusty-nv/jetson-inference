![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/841b9209217f74e5992b8d332c612126)
# Deploying Deep Learning
Welcome to NVIDIA's deep learning inference workshop and end-to-end realtime object recognition library for Jetson TX1.

Included in this repo are resources for efficiently deploying neural networks using NVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)**.  

Vision primitives, such as [`imageNet`](imageNet.h) for image recognition and [`detectNet`](detectNet.h) for object localization, inherit from the shared [`tensorNet`](tensorNet.h) object.  Examples are provided for streaming from live camera feed and processing images from disk.

### **Ten Steps to Deep Learning**

1. [What's Deep Learning?](#whats-deep-learning)
2. [Getting TensorRT](#getting-tensorrt)
3. [Building from Source](#building-from-source)
4. [Digging Into the Code](#digging-into-the-code)
5. [Classifying Images with ImageNet](#classifying-images-with-imagenet)
6. [Running the Live Camera Recognition Demo](#running-the-live-camera-recognition-demo)
7. [Re-training the Network with Customized Data](#re-training-the-network-with-customized-data)
8. [Locating Object Coordinates using DetectNet](#locating-object-coordinates-using-detectNet)
9. [Running the Live Camera Detection Demo](#running-the-live-camera-detection-demo)
10. [Re-training DetectNet with DIGITS](#re-training-detectnet-with-digits)
* [Extra Resources](#extra-resources)
	* [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-institute)
	* [Introductory QwikLabs](https://developer.nvidia.com/deep-learning-courses)
    * [Building nvcaffe](docs/building-nvcaffe.md)
	* [Other Examples](docs/other-examples.md)

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against 
>        JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS)

>  Other branches: [JetPack 2.2 / L4T R24.1 aarch64 (Ubuntu 14.04 LTS)](http://github.com/dusty-nv/jetson-inference/tree/L4T-R24.1) 


## What's Deep Learning?

New to deep neural networks (DNNs) and machine learning?  Take this [introductory primer](docs/deep-learning.md) on training and inference.

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/7aca8779d265a860d5133cdc8c6c6b76" width="800"></a>

Using NVIDIA deep learning tools, it's easy to **[Get Started](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)** training your own DNNs and implementing advanced AI.

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/5720072a6941032685ea18c4e4068a23" width="700"></a>

NVIDIA [DIGITS](https://github.com/NVIDIA/DIGITS) is used to interactively train network models on annotated datasets in the cloud or PC, while TensorRT and Jetson are used to deploy runtime inference in the field.  Together, DIGITS and TensorRT form an effective workflow for developing and deploying deep neural networks capable of advanced AI.  
To get started, see the DIGITS [Getting Started](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md) guide and then the next section of the tutorial, [Getting TensorRT](#getting-tensorrt).

## Getting TensorRT

NVIDIA TensorRT is a new library available in **[JetPack 2.3](https://developer.nvidia.com/embedded/jetpack)** for optimizing and deploying production DNN's.  TensorRT performs a host of graph optimizations and takes advantage of half-precision FP16 support on TX1 to achieve up to 2X or more performance improvement versus Caffe:

<a href="https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/91d88749a582e884926686f7a9a7f9fd" width="700"></a>

And in a benchmark conducted measuring images/sec/Watts, with TensorRT Jetson TX1 is up to 20X more power efficient than traditional CPUs at deep-learning inference.  See this **[Parallel ForAll](https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/)** article for a technical overview of the release.

<a href="https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/86d79898dbb3c0664ab1fcf112da4e6e" width="700"></a>

To obtain TensorRT, download the latest [JetPack](https://developer.nvidia.com/embedded/jetpack) to your PC and re-flash your Jetson (see [Jetson TX1 User Guide](http://developer.nvidia.com/embedded/dlc/l4t-24-1-jetson-tx1-user-guide)).

## Building from Source
Provided along with this repo are TensorRT-enabled examples of running Googlenet/Alexnet on live camera feed for image recognition, and pedestrian detection networks with localization capabilities (i.e. that provide bounding boxes). 

The latest source can be obtained from [GitHub](http://github.com/dusty-nv/jetson-inference) and compiled onboard Jetson TX1.

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against 
>        JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS)
      
#### 1. Cloning the repo
To obtain the repository, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed locally:

``` bash
sudo apt-get install git cmake
```

Then clone the jetson-inference repo:
``` bash
git clone http://github.org/dusty-nv/jetson-inference
```

#### 2. Configuring

When cmake is run, a special pre-installation script (CMakePreBuild.sh) is run and will automatically install any dependencies.

``` bash
mkdir build
cd build
cmake ../
```

#### 3. Compiling

Make sure you are still in the jetson-inference/build directory, created above in step #2.

``` bash
cd jetson-inference/build			# omit if pwd is already /build from above
make
```

Depending on architecture, the package will be built to either armhf or aarch64, with the following directory structure:

```
|-build
   \aarch64		    (64-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```

binaries residing in aarch64/bin, headers in aarch64/include, and libraries in aarch64/lib.

## Classifying Images with ImageNet
There are multiple types of deep learning networks available, including recognition, detection/localization, and soon segmentation.  The first deep learning capability to highlight is **image recognition** using an 'imageNet' that's been trained to identify similar objects.

The [`imageNet`](imageNet.h) object accept an input image and outputs the probability for each class.  Having been trained on ImageNet database of **[1000 objects](data/networks/ilsvrc12_synset_words.txt)**, the standard AlexNet and GoogleNet networks are downloaded during [step 2](#configuring) from above.

After building, first make sure your terminal is located in the aarch64/bin directory:

``` bash
$ cd jetson-inference/build/aarch64/bin
```

Then, classify an example image with the [`imagenet-console`](imagenet-console/imagenet-console.cpp) program.  [`imagenet-console`](imagenet-console/imagenet-console.cpp) accepts 2 command-line arguments:  the path to the input image and path to the output image (with the class overlay printed).

``` bash
$ ./imagenet-console orange_0.jpg output_0.jpg
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/8c63ed0975b4c89a4134c320d4e47931"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/8c63ed0975b4c89a4134c320d4e47931" width="700"></a>

``` bash
$ ./imagenet-console granny_smith_1.jpg output_1.jpg
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/b6aea9d50490fbe261420ab940de0efd"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/b6aea9d50490fbe261420ab940de0efd" width="700"></a>

Next, we will use [imageNet](imageNet.h) to classify a live video feed from the Jetson onboard camera.

## Running the Live Camera Recognition Demo

Similar to the last example, the realtime image recognition demo is located in /aarch64/bin and is called [`imagenet-camera`](imagenet-camera/imagenet-camera.cpp).
It runs on live camera stream and depending on user arguments, loads googlenet or alexnet with TensorRT. 
``` bash
$ ./imagenet-camera googlenet           # to run using googlenet
$ ./imagenet-camera alexnet             # to run using alexnet
```

The frames per second (FPS), classified object name from the video, and confidence of the classified object are printed to the openGL window title bar.  By default the application can recognize up to 1000 different types of objects, since Googlenet and Alexnet are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find included in the repo under [data/networks/ilsvrc12_synset_words.txt](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd" width="800"></a>
<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3" width="800"></a>

## Re-training the Network with Customized Data

The existing GoogleNet and AlexNet models that are downloaded by the repo are pre-trained on [1000 classes of objects](data/networks/ilsvrc12_synset_words.txt).

What if you require a new object class to be added to the network, or otherwise require a different organization of the classes?  
Using [NVIDIA DIGITS](http://github.com/NVIDIA/DIGITS), networks can be fine-tuned or re-trained from a pre-exisiting network model.
After installing DIGITS on a PC or in the cloud (such as an AWS instance), see the **[Image Folder Specification](https://github.com/NVIDIA/DIGITS/blob/master/docs/ImageFolderFormat.md)** to learn how to organize the data for your particular application.
Popular training databases with various annotations and labels include [ImageNet](image-net.org), [MS COCO](mscoco.org), and [Google Images](images.google.com) among others.
See here for a crawler script that will download the 1000 original classes, including as many of the original images that are still available online.

Then, while creating the new network model in DIGITS, copy the [GoogleNet prototxt](data/networks/googlenet.prototxt) and specify the existing GoogleNet caffemodel as the DIGITS **Pretrained Model**:

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/610745a8bafae4a5686d45901f5cc6f3"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/610745a8bafae4a5686d45901f5cc6f3" width="800"></a>

The network training should now converge faster than if it were trained from scratch.  After the desired accuracy has been reached, copy the new model checkpoint back over to your Jetson and proceed as before, but now with the added classes available for recognition.

## Locating Object Coordinates using DetectNet
The previous image recognition examples output class probabilities representing the entire input image.   The second deep learning capability to highlight is detecting multiple objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes).  This is performed using a 'detectNet' - or object detection / localization network.

The [`detectNet`](detectNet.h) object accepts as input the 2D image, and outputs a list of coordinates of the detected bounding boxes.  Three example detection network models are are automatically downloaded during the repo [source configuration](#configuring):

1. **ped-100**  (single-class pedestrian detector)
2. **multiped-500**   (multi-class pedestrian + baggage detector)
3. **facenet-120**  (single-class facial recognition detector)

To process test images with [`detectNet`](detectNet.h) and TensorRT, use the [`detectnet-console`](detectnet-console/detectnet-console.cpp) program.  [`detectnet-console`](detectnet-console/detectnet-console.cpp) accepts command-line arguments representing the path to the input image and path to the output image (with the bounding box overlays rendered).  Some test images are included with the repo:

``` bash
$ ./detectnet-console peds-007.png output-7.png
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/eb1066d317406abb66be939e23150ccc"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/eb1066d317406abb66be939e23150ccc" width="900"></a>

To change the network that [`detectnet-console`](detectnet-console/detectnet-console.cpp) uses, modify [`detectnet-console.cpp`](detectnet-console/detectnet-console.cpp) (beginning line 33):
``` c
detectNet* net = detectNet::Create( detectNet::PEDNET_MULTI );	 // uncomment to enable one of these 
//detectNet* net = detectNet::Create( detectNet::PEDNET );
//detectNet* net = detectNet::Create( detectNet::FACENET );
```
Then to recompile, navigate to the `jetson-inference/build` directory and run `make`.
### Multi-class Object Detection
When using the multiped-500 model (`PEDNET_MULTI`), for images containing luggage or baggage in addition to pedestrians, the 2nd object class is rendered with a green overlay.
``` bash
$ ./detectnet-console peds-008.png output-8.png
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/c0c41b17fb6ea05315b64f3ee7cbbb84"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/c0c41b17fb6ea05315b64f3ee7cbbb84" width="900"></a>

## Running the Live Camera Detection Demo

Similar to the previous example, [`detectnet-camera`](detectnet-camera/detectnet-camera.cpp) runs the object detection networks on live video feed from the Jetson onboard camera.  Launch it from command line along with the type of desired network:

``` bash
$ ./detectnet-camera multiped-500        # run using multi-class pedestrian/luggage detector
$ ./detectnet-camera ped-100             # run using original single-class pedestrian detector
$ ./detectnet-camera facenet-120         # run using facial recognition network
$ ./detectnet-camera                     # by default, program will run using multiped-500
```

> **note**:  to achieve maximum performance while running detectnet, increase the Jetson TX1 clock limits by running the script:
>  `sudo ~/jetson_clocks.sh`

## Re-training DetectNet with DIGITS

For a guide to training custom DetectNets, see the **[Object Detection](https://github.com/NVIDIA/DIGITS/tree/digits-4.0/examples/object-detection)** example included in DIGITS version 4.


## Extra Resources

* [Appendix](docs/aux-contents.md)
	* [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-institute)
	* [Introductory QwikLabs](https://developer.nvidia.com/deep-learning-courses)
    * [Building nvcaffe](docs/building-nvcaffe.md)
	* [Other Examples](docs/other-examples.md)
