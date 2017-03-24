/*
 * inference-101
 */

#ifndef __CUDA_RESIZE_H__
#define __CUDA_RESIZE_H__


#include "cudaUtility.h"


/**
 * Function for increasing or decreasing the size of an image on the GPU.
 * @ingroup util
 */
cudaError_t cudaResize( float* input,  size_t inputWidth,  size_t inputHeight,
				    float* output, size_t outputWidth, size_t outputHeight );


/**
 * Function for increasing or decreasing the size of an image on the GPU.
 * @ingroup util
 */
cudaError_t cudaResizeRGBA( float4* input,  size_t inputWidth,  size_t inputHeight,
				        float4* output, size_t outputWidth, size_t outputHeight );


						

#endif

