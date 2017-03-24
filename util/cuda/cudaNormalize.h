/*
 * inference-101
 */

#ifndef __CUDA_NORMALIZE_H__
#define __CUDA_NORMALIZE_H__


#include "cudaUtility.h"


/**
 * Rebase the pixel intensities of an image between two scales.
 * For example, convert an image with values 0.0-255 to 0.0-1.0.
 * @ingroup util
 */
cudaError_t cudaNormalizeRGBA( float4* input,  const float2& input_range,
						 float4* output, const float2& output_range,
						 size_t  width,  size_t height );

#endif

