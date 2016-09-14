/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_YUV_CONVERT_H
#define __CUDA_YUV_CONVERT_H


#include "cudaUtility.h"
#include <stdint.h>


//////////////////////////////////////////////////////////////////////////////////
/// @name RGBA to YUV 4:2:0 planar (I420 & YV12)
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an RGBA uchar4 buffer into YUV I420 planar.
 */
cudaError_t cudaRGBAToI420( uchar4* input, uint8_t* output, size_t width, size_t height );

/**
 * Convert an RGBA uchar4 texture into YUV I420 planar.
 */
cudaError_t cudaRGBAToI420( uchar4* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );

/**
 * Convert an RGBA uchar4 buffer into YUV YV12 planar.
 */
cudaError_t cudaRGBAToYV12( uchar4* input, uint8_t* output, size_t width, size_t height );

/**
 * Convert an RGBA uchar4 texture into YUV YV12 planar.
 */
cudaError_t cudaRGBAToYV12( uchar4* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV 4:2:2 packed (UYVY & YUYV) to RGBA
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a UYVY 422 packed image into RGBA uchar4.
 */
cudaError_t cudaUYVYToRGBA( uchar2* input, uchar4* output, size_t width, size_t height );

/**
 * Convert a UYVY 422 packed image into RGBA uchar4.
 */
cudaError_t cudaUYVYToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height );

/**
 * Convert a YUYV 422 packed image into RGBA uchar4.
 */
cudaError_t cudaYUYVToRGBA( uchar2* input, uchar4* output, size_t width, size_t height );

/**
 * Convert a YUYV 422 packed image into RGBA uchar4.
 */
cudaError_t cudaYUYVToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name UYUV 4:2:2 packed (UYVY & YUYV) to grayscale
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert a UYVY 422 packed image into a uint8 grayscale.
 */
cudaError_t cudaUYVYToGray( uchar2* input, float* output, size_t width, size_t height );

/**
 * Convert a UYVY 422 packed image into a uint8 grayscale.
 */
cudaError_t cudaUYVYToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height );

/**
 * Convert a YUYV 422 packed image into a uint8 grayscale.
 */
cudaError_t cudaYUYVToGray( uchar2* input, float* output, size_t width, size_t height );

/**
 * Convert a YUYV 422 packed image into a uint8 grayscale.
 */
cudaError_t cudaYUYVToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height );

///@}


//////////////////////////////////////////////////////////////////////////////////
/// @name YUV NV12 to RGBA
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Convert an NV12 texture (semi-planar 4:2:0) to ARGB uchar4 format.
 * NV12 = 8-bit Y plane followed by an interleaved U/V plane with 2x2 subsampling.
 */
cudaError_t cudaNV12ToRGBA( uint8_t* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaNV12ToRGBA( uint8_t* input, uchar4* output, size_t width, size_t height );

cudaError_t cudaNV12ToRGBAf( uint8_t* input, size_t inputPitch, float4* output, size_t outputPitch, size_t width, size_t height );
cudaError_t cudaNV12ToRGBAf( uint8_t* input, float4* output, size_t width, size_t height );

/**
 * Setup NV12 color conversion constants.
 * cudaNV12SetupColorspace() isn't necessary for the user to call, it will be
 * called automatically by cudaNV12ToRGBA() with a hue of 0.0.
 * However if you want to setup custom constants (ie with a hue different than 0),
 * then you can call cudaNV12SetupColorspace() at any time, overriding the default.
 */
cudaError_t cudaNV12SetupColorspace( float hue = 0.0f ); 

///@}

#endif

