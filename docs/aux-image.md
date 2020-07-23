<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
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
	* [Converting to Numpy Arrays](#converting-to-numpy-arrays)
	* [Converting from Numpy Arrays](#converting-from-numpy-arrays)

**CUDA Routines**
* [Color Conversion](#color-conversion)
* [Resizing](#resizing)
* [Cropping](#cropping)
* [Normalization](#normalization)
* [Overlay](#overlay)

For examples of using these functions, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py) in addition to the psuedocode below.  Before diving in here, it's recommended to read the previous page on [Camera Streaming and Multimedia](aux-streaming.md) for info about video capture and output, loading/saving images, ect.

## Image Formats

Although the [video streaming](aux-streaming#source-code) APIs and DNN objects (such [`imageNet`](c/imageNet.h), [`detectNet`](c/detectNet.h), and [`segNet`](c/segNet.h)) expect images in RGB/RGBA format, a variety of other formats are defined for sensor acquisition and low-level I/O:  

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

Memory allocated by [`cudaAllocMapped()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaMappedMemory.h) resides in a shared CPU/GPU memory space, so it is accessible from both the CPU and GPU without needing to perform a memory copy between them (hence it is also referred to as ZeroCopy memory).  

Synchronization is required however - so if you want to access an image from the CPU after GPU processing has occurred, call [`cudaDeviceSynchronize()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d) first.  To free the memory in C++, use the [`cudaFreeHost()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969) function.  In Python, the memory will automatically be released by the garbage collector, but you can free it explicitly with the `del` operator.  

Below is Python and C++ psuedocode for allocating/synchronizing/freeing the ZeroCopy memory:

#### Python
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

#### C++
```cpp
#include <jetson-utils/cudaMappedMemory.h>

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

In C++, you can often omit the explicit [`imageFormat`](#image-formats) enum if your pointers are typed as `uchar3/uchar4/float3/float4`.  Below is functionaly equivalent to the allocation above:

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

CUDA images are also subscriptable, meaning you can index them to directly access the pixel data from the CPU:

```python
for y in range(img.height):
	for x in range(img.width):
		pixel = img[y,x]    # returns a tuple, i.e. (r,g,b) for RGB formats or (r,g,b,a) for RGBA formats
		img[y,x] = pixel    # set a pixel from a tuple (tuple length must match the number of channels)
```

> **note:** the Python subscripting index operator is only available if the image was allocated in mapped ZeroCopy memory (i.e. by [`cudaAllocMapped()`](#image-allocation)).  Otherwise, the data is not accessible from the CPU, and an exception will be thrown. 

The indexing tuple used to access an image may take the following forms:

* `img[y,x]` - note the ordering of the `(y,x)` tuple, same as numpy
* `img[y,x,channel]` - only access a particular channel (i.e. 0 for red, 1 for green, 2 for blue, 3 for alpha)
* `img[y*img.width+x]` - flat 1D index, access all channels in that pixel

Although image subscripting is supported, individually accessing each pixel of a large image isn't recommended to do from Python, as it will significantly slow down the application.  Assuming that a GPU implementation isn't available, a better alternative is to use Numpy.

### Converting to Numpy Arrays

You can access a `cudaImage` memory capsule from Numpy by calling `jetson.utils.cudaToNumpy()` on it first.  The underlying memory isn't copied and Numpy will access it directly - so be aware if you change the data in-place through Numpy, it will be changed in the `cudaImage` capsule as well.

For an example of using `cudaToNumpy()`, see the [`cuda-to-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/dev/python/examples/cuda-to-numpy.py) sample from jetson-utils.

Note that OpenCV expects images in BGR colorspace, so if you plan on using the image with OpenCV, you should call `cv2.cvtColor()` with `cv2.COLOR_RGB2BGR` before using it in OpenCV.

### Converting from Numpy Arrays

Let's say you have an image in a Numpy ndarray, perhaps provided by OpenCV.  As a Numpy array, it will only be accessible from the CPU.  You can use `jetson.utils.cudaFromNumpy()` to copy it to the GPU (into shared CPU/GPU ZeroCopy memory).  

For an example of using `cudaFromNumpy()`, see the [`cuda-from-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/dev/python/examples/cuda-from-numpy.py) sample from jetson-utils.

Note that OpenCV images are in BGR colorspace, so if the image is coming from OpenCV, you should call `cv2.cvtColor()` with `cv2.COLOR_BGR2RGB` first.

## Color Conversion

The [`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) function uses the GPU to convert between image formats and colorspaces.  For example, you can convert from RGB to BGR (or vice versa), from YUV to RGB, RGB to grayscale, ect.  You can also change the data type and number of channels (e.g. RGB8 to RGBA32F).  For more info about the different formats available to convert between, see the [Image Formats](#image-formats) section above.

[`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) has the following limitations and unsupported conversions:
* The YUV formats don't support BGR/BGRA or grayscale (RGB/RGBA only)
* YUV NV12, YUYV, YVYU, and UYVY can only be converted to RGB/RGBA (not from)
* Bayer formats can only be converted to RGB8 (`uchar3`) and RGBA8 (`uchar4`)

The following Python/C++ psuedocode loads an image in RGB8, and convert it to RGBA32F (note that this is purely illustrative, since the image can be loaded directly as RGBA32F).  For a more comprehensive example, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py).

#### Python

```python
import jetson.utils

# load the input image (default format is rgb8)
imgInput = jetson.utils.loadImage('my_image.jpg', format='rgb8') # default format is 'rgb8', but can also be 'rgba8', 'rgb32f', 'rgba32f'

# allocate the output as rgba32f, with the same width/height as the input
imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format='rgba32f')

# convert from rgb8 to rgba32f (the formats used for the conversion are taken from the image capsules)
jetson.utils.cudaConvertColor(imgInput, imgOutput)
```

#### C++

```c++
#include <jetson-utils/cudaColorspace.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

uchar3* imgInput = NULL;   // input is rgb8 (uchar3)
float4* imgOutput = NULL;  // output is rgba32f (float4)

int width = 0;
int height = 0;

// load the image as rgb8 (uchar3)
if( !loadImage("my_image.jpg", &imgInput, &width, &height) )
	return false;

// allocate the output as rgba32f (float4), with the same width/height
if( !cudaAllocMapped(&imgOutput, width, height) )
	return false;

// convert from rgb8 to rgba32f
if( CUDA_FAILED(cudaConvertColor(imgInput, IMAGE_RGB8, imgOutput, IMAGE_RGBA32F, width, height)) )
	return false;	// an error or unsupported conversion occurred
```

## Resizing

The [`cudaResize()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaResize.h) function uses the GPU to rescale images to a different size (either downsampled or upsampled).  The following Python/C++ psuedocode loads an image, and resizes it by a certain factor (downsampled by half in the example).  For a more comprehensive example, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py).

#### Python

```python
import jetson.utils

# load the input image
imgInput = jetson.utils.loadImage('my_image.jpg')

# allocate the output, with half the size of the input
imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width * 0.5, 
                                         height=imgInput.height * 0.5, 
                                         format=imgInput.format)

# rescale the image (the dimensions are taken from the image capsules)
jetson.utils.cudaResize(imgInput, imgOutput)
```

#### C++

```c++
#include <jetson-utils/cudaResize.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

// load the input image
uchar3* imgInput = NULL;

int inputWidth = 0;
int inputHeight = 0;

if( !loadImage("my_image.jpg", &imgInput, &inputWidth, &inputHeight) )
	return false;

// allocate the output image, with half the size of the input
uchar3* imgOutput = NULL;

int outputWidth = inputWidth * 0.5f;
int outputHeight = inputHeight * 0.5f;

if( !cudaAllocMapped(&imgOutput, outputWidth, outputHeight) )
	return false;

// rescale the image
if( CUDA_FAILED(cudaResize(imgInput, inputWidth, inputHeight, imgOutput, outputWidth, outputHeight)) )
	return false;
```

## Cropping

The [`cudaCrop()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaCrop.h) function uses the GPU to crop an images to a particular region of interest (ROI).  The following Python/C++ psuedocode loads an image, and crops it around the center half of the image.  For a more comprehensive example, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py).

Note that the ROI rectangles are provided as `(left, top, right, bottom)` coordinates.

#### Python

```python
import jetson.utils

# load the input image
imgInput = jetson.utils.loadImage('my_image.jpg')

# determine the amount of border pixels (cropping around the center by half)
crop_factor = 0.5
crop_border = ((1.0 - crop_factor) * 0.5 * imgInput.width,
               (1.0 - crop_factor) * 0.5 * imgInput.height)

# compute the ROI as (left, top, right, bottom)
crop_roi = (crop_border[0], crop_border[1], imgInput.width - crop_border[0], imgInput.height - crop_border[1])

# allocate the output image, with the cropped size
imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width * crop_factor,
                                         height=imgInput.height * crop_factor,
                                         format=imgInput.format)

# crop the image to the ROI
jetson.utils.cudaCrop(imgInput, imgOutput, crop_roi)
```

#### C++

```c++
#include <jetson-utils/cudaCrop.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

// load the input image
uchar3* imgInput = NULL;

int inputWidth = 0;
int inputHeight = 0;

if( !loadImage("my_image.jpg", &imgInput, &inputWidth, &inputHeight) )
	return false;

// determine the amount of border pixels (cropping around the center by half)
const float crop_factor = 0.5
const int2  crop_border = make_int2((1.0f - crop_factor) * 0.5f * inputWidth,
                                    (1.0f - crop_factor) * 0.5f * inputHeight);

// compute the ROI as (left, top, right, bottom)
const int4 crop_roi = make_int4(crop_border.x, crop_border.y, inputWidth - crop_border.x, inputHeight - crop_border.y);

// allocate the output image, with half the size of the input
uchar3* imgOutput = NULL;

if( !cudaAllocMapped(&imgOutput, inputWidth * crop_factor, inputHeight * cropFactor) )
	return false;

// crop the image
if( CUDA_FAILED(cudaCrop(imgInput, imgOutput, crop_roi, inputWidth, inputHeight)) )
	return false;
```

## Normalization

The [`cudaNormalize()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaNormalize.h) function uses the GPU to change the range of pixel intensities in an image.  For example, convert an image with pixel values between `[0,1]` to have pixel values between `[0,255]`.  Another common range for pixel values is between `[-1,1]`.

> **note:** all of the other functions in jetson-inference and jetson-utils expect images with pixel ranges between `[0,255]`, so you wouldn't ordinarily need to use `cudaNormalize()`, but it is available in case you are working with data from an alternative source or destination.

The following Python/C++ psuedocode loads an image, and normalizes it from `[0,255]` to `[0,1]`.

#### Python

```python
import jetson.utils

# load the input image (its pixels will be in the range of 0-255)
imgInput = jetson.utils.loadImage('my_image.jpg')

# allocate the output image, with the same dimensions as input
imgOutput = jetson.utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format=imgInput.format)

# normalize the image from [0,255] to [0,1]
jetson.utils.cudaNormalize(imgInput, (0,255), imgOutput, (0,1))
```

#### C++

```c++
#include <jetson-utils/cudaNormalize.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

uchar3* imgInput = NULL;
uchar3* imgOutput = NULL;

int width = 0;
int height = 0;

// load the input image (its pixels will be in the range of 0-255)
if( !loadImage("my_image.jpg", &imgInput, &width, &height) )
	return false;

// allocate the output image, with the same dimensions as input
if( !cudaAllocMapped(&imgOutput, width, height) )
	return false;

// normalize the image from [0,255] to [0,1]
CUDA(cudaNormalize(imgInput, make_float2(0,255),
                   imgOutput, make_float2(0,1),
                   width, height));
```

## Overlay

The [`cudaOverlay()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaOverlay.h) function uses the GPU to compost an input image on top of an output image at a particular location.  Overlay operations are typically called in sequence to form a composite of multiple images together.

The following Python/C++ psuedocode loads two images, and composts them together side-by-side in an output image.

#### Python

```python
import jetson.utils

# load the input images
imgInputA = jetson.utils.loadImage('my_image_a.jpg')
imgInputB = jetson.utils.loadImage('my_image_b.jpg')

# allocate the output image, with dimensions to fit both inputs side-by-side
imgOutput = jetson.utils.cudaAllocMapped(width=imgInputA.width + imgInputB.width, 
                                         height=max(imgInputA.height, imgInputB.height),
                                         format=imgInputA.format)

# compost the two images (the last two arguments are x,y coordinates in the output image)
jetson.utils.cudaOverlay(imgInputA, imgOutput, 0, 0)
jetson.utils.cudaOverlay(imgInputB, imgOutput, imgInputA.width, 0)
```

#### C++

```c++
#include <jetson-utils/cudaOverlay.h>
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

#include <algorithm>  // for std::max()

uchar3* imgInputA = NULL;
uchar3* imgInputB = NULL;
uchar3* imgOutput = NULL;

int2 dimsA = make_int2(0,0);
int2 dimsB = make_int2(0,0);

// load the input images
if( !loadImage("my_image_a.jpg", &imgInputA, &dimsA.x, &dimsA.y) )
	return false;

if( !loadImage("my_image_b.jpg", &imgInputB, &dimsB.x, &dimsB.y) )
	return false;

// allocate the output image, with dimensions to fit both inputs side-by-side
const int2 dimsOutput = make_int2(dimsA.x + dimsB.x, std::max(dimsA.y, dimsB.y));

if( !cudaAllocMapped(&imgOutput, dimsOutput.x, dimsOutput.y) )
	return false;

// compost the two images (the last two arguments are x,y coordinates in the output image)
CUDA(cudaOverlay(imgInputA, dimsA, imgOutput, dimsOutput, 0, 0));
CUDA(cudaOverlay(imgInputB, dimsB, imgOutput, dimsOutput, dimsA.x, 0));
```

##
<p align="right">Back | <b><a href="aux-streaming.md">Camera Streaming and Multimedia</a></p>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

