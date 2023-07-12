<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="aux-streaming.md">Back</a> | <a href="https://github.com/dusty-nv/ros_deep_learning">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Appendix</sup></p>  

# Image Manipulation with CUDA

This page covers a number of image format, conversion, and pre/post-processing functions implemented by jetson-utils with CUDA:

**Image Management**
* [Image Formats](#image-formats)
* [Image Allocation](#image-allocation)
* [Copying Images](#copying-images)
* [Image Capsules in Python](#image-capsules-in-python)
	* [Array Interfaces](#array-interfaces)
	* [Accessing Image Data in Python](#accessing-image-data-in-python)
	* [Accessing as a Numpy Array](#accessing-as-a-numpy-array)
	* [CUDA Array Interface](#cuda-array-interface)
	* [Sharing the Memory Pointer](#sharing-the-memory-pointer)

**CUDA Routines**
* [Color Conversion](#color-conversion)
* [Resizing](#resizing)
* [Cropping](#cropping)
* [Normalization](#normalization)
* [Overlay](#overlay)
* [Drawing Shapes](#drawing-shapes)

For examples of using these functions, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py) in addition to the psuedocode below.  Before diving in here, it's recommended to read the previous page on [Camera Streaming and Multimedia](aux-streaming.md) for info about video capture and output, loading/saving images, ect.

## Image Formats

Although the [video streaming](aux-streaming#source-code) APIs and DNN objects (such [`imageNet`](c/imageNet.h), [`detectNet`](c/detectNet.h), and [`segNet`](c/segNet.h)) expect images in RGB/RGBA format, a variety of other formats are defined for sensor acquisition and low-level I/O:  

|                 | Format string | [`imageFormat` enum](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/group__imageFormat.html#ga931c48e08f361637d093355d64583406)   | Data Type | Bit Depth |
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

To allocate empty GPU memory for storing intermediate/output images (i.e. working memory during processing), use one [`cudaAllocMapped()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaMappedMemory.h) from C++ or the [`cudaImage`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html#cudaImage) object from Python.  Note that the [`videoSource`](aux-streaming#source-code) input streams automatically allocate their own GPU memory, and return to you the latest image, so you needn't allocate your own memory for those.  

Memory allocated by [`cudaAllocMapped()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaMappedMemory.h) resides in a shared CPU/GPU memory space, so it is accessible from both the CPU and GPU without needing to perform a memory copy between them (hence it is also referred to as ZeroCopy memory).  

Synchronization is required however - so if you want to access an image from the CPU after GPU processing has occurred, call [`cudaDeviceSynchronize()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d) first.  To free the memory in C++, use the [`cudaFreeHost()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g71c078689c17627566b2a91989184969) function.  In Python, the memory will automatically be released by the garbage collector, but you can free it explicitly with the `del` operator.  

Below is Python and C++ psuedocode for allocating/synchronizing/freeing the ZeroCopy memory:

#### Python
```python
from jetson_utils import cudaImage, cudaDeviceSynchronize

# allocate a 1920x1080 image in rgb8 format
img = cudaImage(width=1920, height=1080, format='rgb8')

# do some processing on the GPU here
...

# wait for the GPU to finish processing
cudaDeviceSynchronize()

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

## Copying Images

[`cudaMemcpy()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8) can be used to copy memory between images of the same format and dimensions.  It's a standard CUDA function in C++, and there is a similar version for Python in the jetson_utils library:

#### Python
```python
from jetson_utils import cudaMemcpy, cudaImage, loadImage

# load an image and allocate memory to copy it to
img_a = loadImage("my_image.jpg")
img_b = cudaImage(like=img_a)  # short for specifying width, height, format

# copy the image (dst, src)
cudaMemcpy(img_b, img_a)

# or you can use this shortcut, which will return a duplicate
img_c = cudaMemcpy(img_a)
```

#### C++
```cpp
#include <jetson-utils/cudaMappedMemory.h>
#include <jetson-utils/imageIO.h>

uchar3* img_a = NULL;
uchar3* img_b = NULL;

int width = 0;
int height = 0;

// load example image
if( !loadImage("my_image.jpg", &img_a, &width, &height) )
	return false;	// loading error
	
// allocate memory to copy it to
if( !cudaAllocMapped(&img_b, width, height) )
	return false;  // memory error
	
// copy the image (dst, src)
if( CUDA_FAILED(cudaMemcpy(img_b, img_a, width * height * sizeof(uchar3), cudaMemcpyDeviceToDevice)) )
	return false;  // memcpy error
```

## Image Capsules in Python

When you allocate an image in Python, or capture an image from a video feed with [`videoSource.Capture()`](aux-streaming#source-code), it will return a self-contained memory capsule object (of type [`<jetson_utils.cudaImage>`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html#cudaImage)) that can be passed around without having to copy the underlying memory.  

The [`cudaImage`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html#cudaImage) object has the following members:

```python
<cudaImage object>
  .ptr       # memory address (not typically used)
  .size      # size in bytes
  .shape     # (height,width,channels) tuple
  .width     # width in pixels
  .height    # height in pixels
  .channels  # number of color channels
  .format    # format string
  .mapped    # true if ZeroCopy
  .timestamp # timestamp in nanoseconds
```

So you can do things like `img.width` and `img.height` to access properties about the image.

### Array Interfaces

For zero-copy interoperability with other libraries, there exist several ways to access the `cudaImage` memory from Python: 

* [Indexing images directly from Python](#accessing-image-data-in-python)
* [Numpy `__array__` interface](#accessing-as-a-numpy-array), [`cudaToNumpy()`](#converting-to-numpy-arrays), [`cudaFromNumpy()`](#converting-from-numpy-arrays)
* [Numba `__cuda_array_interface__`](#cuda-array-interface) (PyTorch, CuPy, PyCUDA, VPI, ect)
* [Sharing the Memory Pointer](#sharing-the-memory-pointer)

These are implemented so that the underlying memory is mapped and shared with the other libraries as to avoid memory copies.

### Accessing Image Data in Python

CUDA images are subscriptable, meaning you can index them to directly access the pixel data from the CPU:

```python
for y in range(img.height):
    for x in range(img.width):
        pixel = img[y,x]    # returns a tuple, i.e. (r,g,b) for RGB formats or (r,g,b,a) for RGBA formats
        img[y,x] = pixel    # set a pixel from a tuple (tuple length must match the number of channels)
```

> **note:** the Python subscripting index operator is only available if the cudaImage was allocated with `mapped=True` (which is the default).  Otherwise, the data is not accessible from the CPU, and an exception will be thrown. 

The indexing tuple used to access an image may take the following forms:

* `img[y,x]` - note the ordering of the `(y,x)` tuple, same as numpy
* `img[y,x,channel]` - only access a particular channel (i.e. 0 for red, 1 for green, 2 for blue, 3 for alpha)
* `img[y*img.width+x]` - flat 1D index, access all channels in that pixel

Although image subscripting is supported, individually accessing each pixel of a large image isn't recommended to do from Python, as it will significantly slow down the application.  Assuming that a GPU implementation isn't available, a better alternative is to use Numpy.

### Accessing as a Numpy Array

cudaImage supports the Numpy [`__array__`](https://numpy.org/doc/stable/reference/arrays.interface.html) interface protocol, so it can be used in many Numpy functions as if it were a Numpy array without needing to copy it back and forth.  See [`cuda-to-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-numpy.py) for a simple example:

``` python
import numpy as np
from jetson_utils import cudaImage

cuda_img = cudaImage(320, 240, 'rgb32f')
array = np.ones(cuda_img.shape, np.float32)

print(np.add(cuda_img, array))
```

> **note:** Numpy runs on the CPU, so the cudaImage should have been allocated with `mapped=True` (the default) so that it's memory is accessible from both the CPU and GPU.  Any changes to it from Numpy will be reflected in the underlying cudaImage's memory.  

You'll need to use the standalone [numpy routines](https://numpy.org/doc/stable/reference/routines.html) as opposed to the [ndarray class methods](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) (e.g. use `numpy.mean(array) vs array.mean()`) because although cudaImage exports the `__array__` interface for accessing it's memory, it doesn't implement the class methods that Numpy does.  To use all of those, see the [`cudaToNumpy()`](#converting-to-numpy-arrays) function below.

#### Converting to Numpy Arrays

You can explicitly obtain a [`numpy.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) object that's mapped to a `cudaImage` by calling `cudaToNumpy()` on it:  

``` python
import numpy as np
from jetson_utils import cudaImage, cudaToNumpy

cuda_img = cudaImage(320, 240, 'rgb32f')
array = cudaToNumpy(cuda_img)

print(array.mean())
```

As before, the underlying memory isn't copied and Numpy will access it directly - so if you change the data in-place through Numpy, it will be changed in the underlying `cudaImage` as well.  For an example of using `cudaToNumpy()`, see the [`cuda-to-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-numpy.py) sample from jetson-utils.

Note that if you plan on using the image with OpenCV, OpenCV expects images in BGR colorspace, so you should call [`cudaConvertColor()`](#color-conversion) first to convert it from RGB to BGR (see [`cuda-to-cv.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-cv.py) for an example of this).

#### Converting from Numpy Arrays

Let's say you have an image in a Numpy ndarray, perhaps provided by OpenCV - as a Numpy array, it will only be accessible from the CPU.  You can use `cudaFromNumpy()` to copy it to the GPU (into shared CPU/GPU mapped memory).  For an example, see the [`cuda-from-numpy.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-numpy.py) sample:

``` python
import numpy as np
from jetson_utils import cudaFromNumpy

array = np.zeros((240, 320, 3), dtype=np.float32)
cuda_img = cudaFromNumpy(array)
```

Like before if you're using OpenCV, OpenCV images are in BGR colorspace, and you should call [`cudaConvertColor()`](#color-conversion) after to convert it from BGR to RGB (see [`cuda-from-cv.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-cv.py) for an example of this).

### CUDA Array Interface

`cudaImage` also supports Numba's [`__cuda_array_interface__`](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html), which provides zero-copy interoperability among libraries that use GPU memory (including PyTorch, CuPy, PyCUDA, VPI, and others [listed here](https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html#interoperability)).   Similar to how `cudaImage` implements the Numpy [`__array__`](#accessing-as-a-numpy-array) interface, the `__cuda_array_interface__` is transparent to use for passing cudaImage's into Numba/PyTorch/CuPy/PyCUDA functions, and the memory is shared with these libraries so there aren't memory copies or extra overhead.

For an example of this, see the code from [`cuda-array-interface.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-array-interface.py):

``` python
import cupy
from jetson_utils import cudaImage

cuda_img = cudaImage(640, 480, 'rgb8')
cupy_array = cupy.ones((480, 640, 3))

print(cupy.add(cuda_img, cupy_array))
```

> **note:** `cudaImage` also implements [PyCUDA's `gpudata` interface](https://documen.tician.de/pycuda/array.html) as well and can be used like a PyCUDA `GPUArray`.

Another example can be found in [`cuda-to-pytorch.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-to-pytorch.py) for sharing memory with PyTorch tensors:

``` python
import torch
from jetson_utils import cudaImage

# allocate cuda memory
cuda_img = cudaImage(640, 480, 'rgb8')

# map to torch tensor using __cuda_array_interface__
tensor = torch.as_tensor(cuda_img, device='cuda')
```

This enables the GPU memory from the cudaImage to be used by PyTorch GPU tensors without copying it - any changes that PyTorch makes to contents of the tensor will be reflected in the cudaImage.  Note that `device='cuda'` should be specified in order for PyTorch to perform the zero-copy mapping - you can check this by confirming that the data pointers match between the cudaImage and PyTorch tensor objects.

### Sharing the Memory Pointer

For libraries that don't support one of the above interfaces, cudaImage exposes the raw data pointer of it's memory through it's `.ptr` attribute, which can be used to import it into other data structures without copying it.  Conversely, the cudaImage initializer also has a `ptr` argument that can be set to an externally-allocated buffer - in this case, cudaImage will share the memory instead allocating it's own.

See [`cuda-from-pytorch.py`](https://github.com/dusty-nv/jetson-utils/blob/master/python/examples/cuda-from-pytorch.py) for an example of doing this, where an existing PyTorch GPU tensor is mapped to a cudaImage:

``` python
import torch
from jetson_utils import cudaImage

# allocate a GPU tensor with NCHW layout (strided colors)
tensor = torch.rand(1, 3, 480, 640, dtype=torch.float32, device='cuda')

# transpose the channels to NHWC layout (interleaved colors)
tensor = tensor.to(memory_format=torch.channels_last)   # or tensor.permute(0, 3, 2, 1)

# map to cudaImage using the same underlying memory (any changes will be reflected in the PyTorch tensor)
cuda_img = cudaImage(ptr=tensor.data_ptr(), width=tensor.shape[-1], height=tensor.shape[-2], format='rgb32f')
```

> **note:** be aware of NCHW [channel layout](https://pytorch.org/blog/tensor-memory-format-matters/#memory-formats-supported-by-pytorch-operators) (strided colors) vs. NHWC layout (interleaved colors), as cudaImage expects the later.

When external pointers are mapped into a cudaImage, by default the cudaImage does not take ownership over the underlying memory and will not free it when the cudaImage is released (to change this, set `freeOnDelete=True` in the initializer).  Handling synchronization between libraries should be implemented by the user (e.g. so that PyTorch isn't accessing the memory at the same time that the cudaImage is used). 

## Color Conversion

The [`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) function uses the GPU to convert between image formats and colorspaces.  For example, you can convert from RGB to BGR (or vice versa), from YUV to RGB, RGB to grayscale, ect.  You can also change the data type and number of channels (e.g. RGB8 to RGBA32F).  For more info about the different formats available to convert between, see the [Image Formats](#image-formats) section above.

[`cudaConvertColor()`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaColorspace.h) has the following limitations and unsupported conversions:
* The YUV formats don't support BGR/BGRA or grayscale (RGB/RGBA only)
* YUV NV12, YUYV, YVYU, and UYVY can only be converted to RGB/RGBA (not from)
* Bayer formats can only be converted to RGB8 (`uchar3`) and RGBA8 (`uchar4`)

The following Python/C++ psuedocode loads an image in RGB8, and convert it to RGBA32F (note that this is purely illustrative, since the image can be loaded directly as RGBA32F).  For a more comprehensive example, see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py).

#### Python

```python
import jetson_utils

# load the input image (default format is rgb8)
imgInput = jetson_utils.loadImage('my_image.jpg', format='rgb8') # default format is 'rgb8', but can also be 'rgba8', 'rgb32f', 'rgba32f'

# allocate the output as rgba32f, with the same width/height as the input
imgOutput = jetson_utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format='rgba32f')

# convert from rgb8 to rgba32f (the formats used for the conversion are taken from the image capsules)
jetson_utils.cudaConvertColor(imgInput, imgOutput)
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
import jetson_utils

# load the input image
imgInput = jetson_utils.loadImage('my_image.jpg')

# allocate the output, with half the size of the input
imgOutput = jetson_utils.cudaAllocMapped(width=imgInput.width * 0.5, 
                                         height=imgInput.height * 0.5, 
                                         format=imgInput.format)

# rescale the image (the dimensions are taken from the image capsules)
jetson_utils.cudaResize(imgInput, imgOutput)
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
import jetson_utils

# load the input image
imgInput = jetson_utils.loadImage('my_image.jpg')

# determine the amount of border pixels (cropping around the center by half)
crop_factor = 0.5
crop_border = ((1.0 - crop_factor) * 0.5 * imgInput.width,
               (1.0 - crop_factor) * 0.5 * imgInput.height)

# compute the ROI as (left, top, right, bottom)
crop_roi = (crop_border[0], crop_border[1], imgInput.width - crop_border[0], imgInput.height - crop_border[1])

# allocate the output image, with the cropped size
imgOutput = jetson_utils.cudaAllocMapped(width=imgInput.width * crop_factor,
                                         height=imgInput.height * crop_factor,
                                         format=imgInput.format)

# crop the image to the ROI
jetson_utils.cudaCrop(imgInput, imgOutput, crop_roi)
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
import jetson_utils

# load the input image (its pixels will be in the range of 0-255)
imgInput = jetson_utils.loadImage('my_image.jpg')

# allocate the output image, with the same dimensions as input
imgOutput = jetson_utils.cudaAllocMapped(width=imgInput.width, height=imgInput.height, format=imgInput.format)

# normalize the image from [0,255] to [0,1]
jetson_utils.cudaNormalize(imgInput, (0,255), imgOutput, (0,1))
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
import jetson_utils

# load the input images
imgInputA = jetson_utils.loadImage('my_image_a.jpg')
imgInputB = jetson_utils.loadImage('my_image_b.jpg')

# allocate the output image, with dimensions to fit both inputs side-by-side
imgOutput = jetson_utils.cudaAllocMapped(width=imgInputA.width + imgInputB.width, 
                                         height=max(imgInputA.height, imgInputB.height),
                                         format=imgInputA.format)

# compost the two images (the last two arguments are x,y coordinates in the output image)
jetson_utils.cudaOverlay(imgInputA, imgOutput, 0, 0)
jetson_utils.cudaOverlay(imgInputB, imgOutput, imgInputA.width, 0)
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

## Drawing Shapes

[`cudaDraw.h`](https://github.com/dusty-nv/jetson-utils/tree/master/cuda/cudaDraw.h) defines several functions for drawing basic shapes, including circles, lines, and rectangles.

Below are simple Python and C++ psuedocode for using them - see [`cuda-examples.py`](https://github.com/dusty-nv/jetson-utils/tree/master/python/examples/cuda-examples.py) for a functioning example.

#### Python

``` python
# load the input image
input = jetson_utils.loadImage("my_image.jpg")

# cudaDrawCircle(input, (cx,cy), radius, (r,g,b,a), output=None)
jetson_utils.cudaDrawCircle(input, (50,50), 25, (0,255,127,200))

# cudaDrawRect(input, (left,top,right,bottom), (r,g,b,a), output=None)
jetson_utils.cudaDrawRect(input, (200,25,350,250), (255,127,0,200))

# cudaDrawLine(input, (x1,y1), (x2,y2), (r,g,b,a), line_width, output=None)
jetson_utils.cudaDrawLine(input, (25,150), (325,15), (255,0,200,200), 10)
```

> **note:** if the optional `output` image isn't specified, the operation will be performed in-place on the `input` image.

#### C++

``` cpp
#include <jetson-utils/cudaDraw.h>
#include <jetson-utils/imageIO.h>

uchar3* img = NULL;
int width = 0;
int height = 0;

// load example image
if( !loadImage("my_image.jpg", &img, &width, &height) )
	return false;	// loading error
	
// see cudaDraw.h for definitions
CUDA(cudaDrawCircle(img, width, height, 50, 50, 25, make_float4(0,255,127,200)));
CUDA(cudaDrawRect(img, width, height, 200, 25, 350, 250, make_float4(255,127,0,200)));
CUDA(cudaDrawLine(img, width, height, 25, 150, 325, 15, make_float4(255,0,200,200), 10));
```

##
<p align="right">Next | <b><a href="https://github.com/dusty-nv/ros_deep_learning">Deep Learning Nodes for ROS/ROS2</a></b>
<br/>
Back | <b><a href="aux-streaming.md">Camera Streaming and Multimedia</a></p>
<p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>

