

/**
 * @defgroup deepVision DNN Vision Library (jetson-inference)
 * C++ runtime library for supporting deep vision neural network models and inferencing with TensorRT.
 * Available network primitives currently include:  
 *   - \ref imageNet for image recognition
 *   - \ref detectNet for object detection + localization
 *   - \ref segNet for semantic segmentation
 *   - \ref poseNet for pose estimation
 *   - \ref depthNet for mono depth
 *
 * The different primitives each inherit from the shared \ref tensorNet object.
 *
 * @see \ref util for various utilities, including \ref cuda functions and \ref video interfaces.
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
 * @defgroup poseNet poseNet
 * Pose estimation DNN
 * @ingroup deepVision
 */

/**
 * @defgroup actionNet actionNet
 * Action/activity recognition DNN
 * @ingroup deepVision
 */
 
/**
 * @defgroup depthNet depthNet
 * Mono depth estimation from monocular images.
 * @ingroup deepVision
 */

/**
 * @defgroup backgroundNet backgroundNet
 * Foreground/background segmentation and removal DNN
 * @ingroup deepVision
 */
 
/**
 * @defgroup objectTracker Object Tracking
 * Object tracking used by detectNet
 * @ingroup deepVision
 */
 
/**
 * @defgroup modelDownloader Model Downloader
 * Utilities for automatically downloading pre-trained models.
 * @ingroup deepVision
 */
 
/**
 * @defgroup util Utilities Library (jetson-utils)
 * Tools and utilities for video streaming, codecs, display, and CUDA processing.
 */

/**
 * @defgroup camera Camera Capture
 * Camera capture and streaming for MIPI CSI and V4L2 devices (deprecated).
 * @deprecated See the updated \ref video APIs.
 * @ingroup util
 */

/**
 * @defgroup codec Codec
 * Hardware-accelerated video encoder and decoder (H.264/H.265) using GStreamer.
 * @ingroup util
 */

/**
 * @defgroup commandLine Command-Line Parsing
 * Parsing of command line arguments, flags, and variables.
 * @ingroup util
 */

/**
 * @defgroup csv CSV Parsing
 * Text file parsing for Comma-Separated Value (CSV) formats, with user-defined
 * delimiters beyond just commas, including spaces, tabs, or other symbols.
 * @ingroup util
 */

/**
 * @defgroup cuda CUDA
 * CUDA utilities and image processing kernels.
 * @ingroup util
 */

/**
 * @defgroup colormap Color Mapping
 * Defines various colormaps and color mapping functions.
 * @ingroup cuda
 */

/**
 * @defgroup colorspace Color Conversion
 * Colorspace conversion functions for various YUV formats, RGB, BGR, Bayer, and grayscale.
 * @see cudaConvertColor for automated format conversion.
 * @see cudaYUV.h for the YUV functions.
 * @ingroup cuda
 */

/**
 * @defgroup crop Cropping
 * Crop an image to the specified region of interest (ROI).
 * @ingroup cuda
 */

/**
 * @defgroup drawing Drawing
 * Drawing basic 2D shapes using CUDA.
 * @ingroup cuda
 */
 
/**
 * @defgroup cudaError Error Checking
 * Error checking and logging macros.
 * @ingroup cuda
 */

/**
 * @defgroup cudaFont Fonts
 * TTF font rasterization and image overlay rendering using CUDA.
 * @ingroup cuda
 */

/**
 * @defgroup cudaMemory Memory Management
 * Allocation of CUDA mapped zero-copy memory.
 * @ingroup cuda
 */
 
/**
 * @defgroup normalization Normalization
 * Normalize the pixel intensities of an image between two ranges.
 * @ingroup cuda
 */

/**
 * @defgroup overlay Overlay
 * Overlay images and vector shapes onto other images.
 * @ingroup cuda
 */

/**
 * @defgroup cudaFilter Pixel Filtering
 * CUDA device functions for sampling pixels with bilinear filtering.
 * @ingroup cuda
 */

/**
 * @defgroup pointCloud Point Cloud
 * 3D point cloud processing and visualization.
 * @ingroup cuda
 */

/**
 * @defgroup resize Resize
 * Rescale an image to a different resolution.
 * @ingroup cuda
 */

/**
 * @defgroup warping Warping
 * Various image warps and matrix transforms.
 * @ingroup cuda
 */
 
/**
 * @defgroup filesystem Filesystem
 * Functions for listing files in directories and manipulating file paths.
 * @ingroup util
 */

/**
 * @defgroup imageFormat Image Formats
 * Enumerations and helper functions for different types of image formats.
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
 * @defgroup log Logging
 * Message logging with a variable level of output and destinations.
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
 * @defgroup network Networking
 * TCP/UDP sockets and IP address manipulation.
 * @ingroup util
 */
 
/**
 * @defgroup OpenGL OpenGL
 * OpenGL textures and display window (X11/GLX).
 * @ingroup util
 */

/**
 * @defgroup time Time
 * Timestamping operations for measuring the timing of CPU code.
 * @ingroup util
 */

/**
 * @defgroup video Video Streaming
 * videoSource and videoOutput APIs for input and output video streams.
 * @ingroup util
 */

