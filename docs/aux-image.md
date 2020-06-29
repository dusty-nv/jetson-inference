<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="aux-streaming.md">Back</a> | <a href="../README.md#hello-ai-world">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Appendix</sup></p>  

# Image Manipulation with CUDA

This page covers a number of image format, conversion, and pre/post-processing functions implemented by jetson-utils with CUDA:

**Image Management**
* [Image Formats](#image-formats)
* [Image Allocation](#image-allocation)
* [Image Capsules in Python](#image-capsules-in-python)
	* [Accessing Image Data in Python](#accessing-image-data-in-python)
	* [Converting from Numpy Arrays](#converting-from-numpy-arrays)
	* [Converting to Numpy Arrays](#converting-to-numpy-arrays)

**CUDA routines**
* [Color Conversion](#color-conversion)

Unless you're customizing your own application or video interface, you may not typically encounter the topics covered below. Before diving in here, please see the previous page on [Camera Streaming and Multimedia](aux-streaming.md) for info about video capture and output, loading/saving images, ect.

## Image Formats

Although the [video streaming](aux-streaming#source-code) APIs and DNN objects (such [`imageNet`](c/imageNet.h), [`detectNet`](c/detectNet.h), and [`segNet`](c/segNet.h)) expect images in RGB/RGBA format, a variety of other formats are defined for low-level I/O.  

|                 | Format string | [`imageFormat` enum](https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/group__imageFormat.html#ga931c48e08f361637d093355d64583406)   | Data Type | Bit Depth |
|-----------------|---------------|--------------------|-----------|-----------|
| **RGB/RGBA**    | `rgb8`        | `IMAGE_RGB8`       | `uchar3`  | 24        |
|                 | `rgba8`       | `IMAGE_RGBA8`      | `uchar4`  | 32        |
|                 | `rgb32f`      | `IMAGE_RGB32F`     | `float3`  | 96        |
|                 | `rgba32f`     | `IMAGE_RGBA32F`    | `float4`  | 128       |
| **BGR/BGRA**    | `bgr8`        | `IMAGE_BGR8`       | `uchar3`  | 24        |
|                 | `bgra8`       | `IMAGE_BGRA8`      | `uchar4`  | 32        |
|                 | `bgr32f`      | `IMAGE_BGR32F`     | `float3`  | 96        |
|                 | `bgra32f`     | `IMAGE_BGRA32F`    | `float4`  | 128       |
| **YUV (4:2:2)** | `yuyv`        | `IMAGE_YUYV`       | `uint8`   | 16        |
|                 | `yuy2`        | `IMAGE_YUY2`       | `uint8`   | 16        |
|                 | `yvyu`        | `IMAGE_YVYU`       | `uint8`   | 16        |
|                 | `uyvy`        | `IMAGE_UYVY`       | `uint8`   | 16        |
| **YUV (4:2:0)** | `i420`        | `IMAGE_I420`       | `uint8`   | 12        |
|                 | `yv12`        | `IMAGE_YV12`       | `uint8`   | 12        |
|                 | `nv12`        | `IMAGE_NV12`       | `uint8`   | 12        |
| **Bayer**       | `bayer-bggr`  | `IMAGE_BAYER_BGGR` | `uint8`   | 8         |
|                 | `bayer-gbrg`  | `IMAGE_BAYER_GBRG` | `uint8`   | 8         |
|                 | `bayer-grbg`  | `IMAGE_BAYER_GRBG` | `uint8`   | 8         |
|                 | `bayer-rggb`  | `IMAGE_BAYER_RGGB` | `uint8`   | 8         |
| **Grayscale**   | `gray8`       | `IMAGE_GRAY8`      | `uint8`   | 8         |
|                 | `gray32f`     | `IMAGE_GRAY32F`    | `float`   | 32        |
* The bit depth represents the effective number of bits per pixel
* For detailed specifications of the YUV formats, refer to [fourcc.org](http://fourcc.org/yuv.php)

> **note:** in C++, the RGB/RGBA formats are the only ones that should be used with the `uchar3`/`uchar4`/`float3`/`float4` vector types.  It is assumed that when these types are used, the images are in RGB/RGBA format.

To convert images between data formats and/or colorspaces, see the [Color Conversion](#color-conversion) section below.

## Image Allocation

To allocate empty GPU memory for storing intermediate/output images (i.e. working memory during processing), use one of the [`cudaAllocMapped()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaMappedMemory.h) functions from C++ or Python.  Note that the [`videoSource`](aux-streaming#source-code) input streams automatically allocate their own GPU memory, and return to you the latest image, so you needn't allocate your own memory for those.  

Memory allocated by [`cudaAllocMapped()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaMappedMemory.h) resides in a shared CPU/GPU memory space, so it is accessible from both the CPU and GPU without needing to perform a memory copy between them (hence it is also referred to as ZeroCopy memory).  Synchronization is required however - so if you want to access an image from the CPU after GPU processing has occurred, call [`cudaDeviceSynchronize()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d) first.

In C++, use [`cudaFreeHost()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969) to free the memory.  In Python, the memory will automatically be released by the garbage collector, but you can do it explicitly with the `del` operator.  

Below is Python and C++ psuedocode for allocating/synchronizing/freeing the ZeroCopy memory:

**Python**
```python
import jetson.utils

# allocate a 1920x1080 image in rgb8 format
img = jetson.utils.cudaAllocMapped(width=1920, height=1080, format='rgb8')

# do some processing on the GPU here
...

# wait for the GPU to finish processing
jetson.utils.cudaDeviceSynchronize()

# Python will automatically free the memory, but you can explicitly do it with 'del'
del img
```

**C++**
```cpp
#include <jetson-utils/cudaAllocMapped.h>

void* img = NULL;

// allocate a 1920x1080 image in rgb8 format
if( !cudaAllocMapped(&img, 1920, 1080, IMAGE_RGB8) )
	return false;	// memory error

// do some processing on the GPU here 
...

// wait for the GPU to finish processing
CUDA(cudaDeviceSynchronize());

// release the memory
CUDA(cudaFreeHost(img));
```

In C++, you can often omit the explicit imageFormat enums if your pointers are typed as `uchar3/uchar4/float3/float4`.  Below is functionaly equivalent to the allocation above:

```cpp
uchar3* img = NULL;	// can be uchar3 (rgb8), uchar4 (rgba8), float3 (rgb32f), float4 (rgba32f)

if( !cudaAllocMapped(&img, 1920, 1080) )
	return false;	
```

> **note:** when using these vector types, these images will be assumed to be in their respective RGB/RGBA colorspace.  So if you use `uchar3/uchar4/float3/float4` to represent an image that contains BGR/BGRA data, it could be intepreted by some processing functions as RGB/RGBA unless you explicitly specify the proper [image format](#image-formats).

## Image Capsules in Python

When you allocate an image in Python, or capture an image from a video feed with [`videoSource.Capture()`](aux-streaming#source-code), it will return a self-contained memory capsule object (of type `<jetson.utils.cudaImage>`) that can be passed around without having to copy the underlying memory.  The `cudaImage` object has the following members:

```python
<jetson.utils.cudaImage>
 .ptr      # memory address (not typically used)
 .size     # size in bytes
 .shape    # (height,width,channels) tuple
 .width    # width in pixels
 .height   # height in pixels
 .channels # number of color channels
 .format   # format string
 .mapped   # true if ZeroCopy
```

So you can do things like `img.width` and `img.height` to access properties about the image.

### Accessing Image Data in Python

CUDA images are also subscriptable, meaning you can index them to access the pixel data:

```python
for y in range(img.height):
	for x in range(img.width):
		pixel = img[y,x]	# returns a tuple, i.e. (r,g,b) for RGB formats or (r,g,b,a) for RGBA formats
		img[y,x] = pixel    # set a pixel from a tuple (tuple length must match the number of channels)
```

> **note:** the Python subscripting index operator is only applicable if the image was allocated in mapped ZeroCopy memory (i.e. by [`cudaAllocMapped()`](#image-allocation)).  Otherwise, the data is not accessible from the CPU, and an exception will be thrown. 

Although possible, individually accessing each pixel of a large image isn't recommended to do from Python, as it will significantly slow down the application.  Assuming that a GPU implementation isn't available, a better alternative is to use Numpy.

### Converting from Numpy Arrays

Let's say you have an image in a Numpy ndarray, perhaps provided by OpenCV.  As a Numpy array, it will only be accessible from the CPU.  You can use `jetson.utils.cudaFromNumpy()` to copy it to the GPU (into shared CPU/GPU ZeroCopy memory).  For an example, see [`cuda-from-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/dev/python/examples/cuda-from-numpy.py) from jetson-utils.

Note that OpenCV images are in BGR colorspace, so if the image is coming from OpenCV, you should call `cv2.cvtColor()` with `cv2.COLOR_BGR2RGB` first.

### Converting to Numpy Arrays

You can access a `cudaImage` memory capsule from Numpy by calling `jetson.utils.cudaToNumpy()` on it first.  In this case, the underlying memory isn't copied and Numpy will access it directly - so be aware if you change the data in-place through Numpy, it will be changed in the `cudaImage` capsule as well.

For an example of using `cudaToNumpy()`, see [`cuda-to-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/dev/python/examples/cuda-to-numpy.py) from jetson-utils.

Note that OpenCV expects images in BGR colorspace, so if you plan on using the image with OpenCV, you should call `cv2.cvtColor()` with `cv2.COLOR_RGB2BGR` before using it in OpenCV.

## Color Conversion

The [`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) function uses the GPU to convert between image formats and colorspaces.  For example, you can convert from RGB to BGR (or vice versa), from YUV to RGB, RGB to grayscale, ect.  You can also change the data type and number of channels (e.g. `rgb8` to `rgba32f`).  For more info about the different formats available, see the [Image Formats](#image-formats) section above.

[`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) is defined in `cudaColorspace.h` for C++, and in Python as `jetson.utils.cudaConvertColor()`.

##
<p align="right">Back | <b><a href="aux-streaming.md">Camera Streaming and Multimedia</a></b>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

