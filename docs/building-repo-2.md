<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="jetpack-setup-2.md">Back</a> | <a href="imagenet-console-2.md">Next</a> | </sup><b><a href="../README.md"><sup>Contents</sup></a></b>
<br/>
<sup>System Setup</sup></p>   

# Building the Repo from Source

Provided along with this repo are TensorRT-enabled deep learning primitives for running Googlenet/Alexnet on live camera feed for image recognition, pedestrian detection networks with localization capabilities (i.e. that provide bounding boxes), and segmentation.  This repo is intended to be built & run on the Jetson and to accept the network models from the host PC trained on the DIGITS server.

The latest source can be obtained from [GitHub](http://github.com/dusty-nv/jetson-inference) and compiled onboard Jetson AGX Xavier and Jetson TX1/TX2.
      
#### Cloning the Repo
To obtain the repository, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed locally:

``` bash
$ sudo apt-get install git cmake
```

Then clone the jetson-inference repo:
``` bash
$ git clone https://github.com/dusty-nv/jetson-inference
$ cd jetson-inference
$ git submodule update --init
```

#### Configuring with CMake

When cmake is run, a special pre-installation script (CMakePreBuild.sh) is run and will automatically install any dependencies.

``` bash
$ mkdir build
$ cd build
$ cmake ../
```

> **note**: the cmake command will launch the CMakePrebuild.sh script which asks for sudo while making sure prerequisite packages have been installed on the Jetson. The script also downloads the network model snapshots from web services.

#### Compiling the Project

Make sure you are still in the jetson-inference/build directory, created above in step #2.

``` bash
$ cd jetson-inference/build			# omit if pwd is already /build from above
$ make
$ sudo make install
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

In the build tree, you can find the binaries residing in `build/aarch64/bin`, headers in `build/aarch64/include`, and libraries in `build/aarch64/lib`.  These also get installed under `/usr` during the `sudo make install` step run above.

#### Digging Into the Code

For reference, see the available vision primitives, including [`imageNet`](imageNet.h) for image recognition and [`detectNet`](detectNet.h) for object localization.

``` c++
/**
 * Image recognition with GoogleNet/Alexnet or custom models, using TensorRT.
 */
class imageNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		ALEXNET,
		GOOGLENET
	};

	/**
	 * Load a new network instance
	 */
	static imageNet* Create( NetworkType networkType=GOOGLENET );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 */
	static imageNet* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
							 const char* class_labels, const char* input="data", const char* output="prob" );

	/**
	 * Determine the maximum likelihood image class.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );
};
```

Both inherit from the shared [`tensorNet`](tensorNet.h) object which contains common TensorRT code.

##
<p align="right">Next | <b><a href="imagenet-console-2.md">Classifying Images with ImageNet</a></b>
<br/>
Back | <b><a href="jetpack-setup-2.md">Setting up Jetson with JetPack</a></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><b><a href="../README.md"><sup>Table of Contents</sup></a></b></p>
