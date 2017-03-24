/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_RGB_CONVERT_H
#define __CUDA_RGB_CONVERT_H


#include "cudaUtility.h"
#include <stdint.h>


/**
 * Convert 8-bit fixed-point RGB image to 32-bit floating-point RGBA image
 * @ingroup util
 */
cudaError_t cudaRGBToRGBAf( uchar3* input, float4* output, size_t width, size_t height );


#endif
