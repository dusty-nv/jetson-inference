

/**
 * @defgroup deepVision DNN Vision Library (jetson-inference)
 * C++ runtime library for supporting deep vision neural network models and inferencing with TensorRT.
 * Available network primitives currently include:  
 *   - \ref imageNet for image recognition
 *   - \ref detectNet for object detection + localization
 *   - \ref segNet for segmentation
 *   - \ref homographyNet for homography estimation
 *   - \ref superResNet for super-resolution upsampling
 *
 * The different primitives each inherit from the shared \ref tensorNet object.
 */


/**
 * @defgroup tensorNet tensorNet
 * DNN abstract base class that provides TensorRT functionality underneath.
 * These functions aren't typically accessed by end users unless
 * they are implementing their own DNN class like imageNet or detectNet.
 * @ingroup deepVision
 */

/**
 * @defgroup imageNet imageNet
 * Image recognition DNN (GoogleNet, AlexNet, ResNet)
 * @ingroup deepVision
 */

/**
 * @defgroup detectNet detectNet
 * Object detection DNN (SSD, DetectNet)
 * @ingroup deepVision
 */

/**
 * @defgroup segNet segNet
 * Semantic segmentation DNN (FCN or Fully-Convolutional Networks)
 * @ingroup deepVision
 */

/**
 * @defgroup homographyNet homographyNet
 * Homography estimation DNN for registering and aligning images.
 * @ingroup deepVision
 */

/**
 * @defgroup superResNet superResNet
 * Super resolution DNN for upscaling images.
 * @ingroup deepVision
 */

/**
 * @defgroup stereoNet stereoNet
 * Stereo disparity depth estimation.
 * @ingroup deepVision
 */

/**
 * @defgroup depthNet depthNet
 * Depth estimation from monocular images.
 * @ingroup deepVision
 */

/**
 * @defgroup flowNet flowNet
 * Dense optical flow estimation DNN.
 * @ingroup deepVision
 */

/**
 * @defgroup util Utilities Library (jetson-utils)
 * Tools and utilities for camera streaming, codecs, display, and visualization.
 */

/**
 * @defgroup camera Camera Capture
 * Camera capture and streaming for MIPI CSI and V4L2 devices.
 * @ingroup util
 */

/**
 * @defgroup gstCamera gstCamera
 * Camera capture and streaming using GStreamer for MIPI CSI and V4L2 devices.
 * @ingroup camera
 */

/**
 * @defgroup codec Codec
 * Hardware-accelerated video encoder and decoder (H.264/H.265) using GStreamer.
 * @ingroup util
 */
 
/**
 * @defgroup cuda CUDA
 * CUDA utilities and image processing kernels.
 * @ingroup util
 */

/**
 * @defgroup cudaFont cudaFont
 * TTF font rasterization and image overlay rendering using CUDA.
 * @ingroup cuda
 */

/**
 * @defgroup colormap Color Mapping
 * Defines various colormaps and color mapping functions.
 * @ingroup cuda
 */

/**
 * @defgroup colorspace Color Conversion
 * Colorspace conversion functions for various YUV formats, RGB, BGR, ect.
 * @ingroup cuda
 */

/**
 * @defgroup warping Warping
 * Various image warps and matrix transforms.
 * @ingroup cuda
 */
 
/**
 * @defgroup OpenGL OpenGL
 * OpenGL textures and display window (X11/GLX).
 * @ingroup util
 */

/**
 * @defgroup image Image I/O
 * Loading and saving image files from disk.
 * @ingroup util
 */
 
/**
 * @defgroup input Input
 * HID input devices including gamepad controllers, joysticks, and keyboard.
 * @ingroup util
 */

/**
 * @defgroup matrix Matrix
 * 3x3 matrix operations from `mat33.h`
 * @ingroup util
 */
 
/**
 * @defgroup threads Multithreading
 * Mutex, events, threads, and process objects based on pthreads.
 * @ingroup util
 */
 
/**
 * @degroup network Networking
 * TCP/UDP sockets and IP address manipulation.
 */
 
/**
 * @defgroup filesystem Filesystem
 * Functions for listing files in directories and manipulating file paths.
 * @ingroup util
 */

/**
 * @defgroup time Time
 * Timestamping operations for measuring the timing of CPU code.
 * @ingroup util
 */

