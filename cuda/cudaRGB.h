/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_RGB_CONVERT_H
#define __CUDA_RGB_CONVERT_H


#include "cudaUtility.h"
#include <stdint.h>


//////////////////////////////////////////////////////////////////////////////////
/// @name RGB to RGBA
//////////////////////////////////////////////////////////////////////////////////

cudaError_t cudaRGBToRGBAf( uint8_t* input, float4* output, size_t width, size_t height );
cudaError_t cudaBAYER_GR8toRGBA( uint8_t* input, float4* output, size_t width, size_t height );


#endif

