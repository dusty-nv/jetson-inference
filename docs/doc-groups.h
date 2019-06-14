

/**
 * @defgroup deepVision DNN Vision Library (jetson-inference)
 * C++ runtime library for supporting deep vision neural network models and inferencing with TensorRT.
 * Available network primitives currently include:  
 *   - \ref imageNet for image recognition
 *   - \ref detectNet for object detection + localization
 *   - \ref segNet for segmentation
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
 * @defgroup cuda CUDA
 * CUDA utilities and image processing kernels.
 * @ingroup util
 */

/**
 * @defgroup cudaFont cudaFont
 * TTF font rasterization and image overlay rendering using CUDA.
 * @ingroup cuda
 */



