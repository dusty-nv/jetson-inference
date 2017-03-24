/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_OVERLAY_H__
#define __CUDA_OVERLAY_H__

#include "cudaUtility.h"


/**
 * cudaRectOutlineOverlay
 * @ingroup util
 */
cudaError_t cudaRectOutlineOverlay( float4* input, float4* output, uint32_t width, uint32_t height, float4* boundingBoxes, int numBoxes, const float4& color );


/**
 * cudaRectFillOverlay
 * @ingroup util
 */
//cudaError_t cudaRectFillOverlay( float4* input, float4* output, uint32_t width, uint32_t height, float4* boundingBoxes, int numBoxes, const float4& color );



#endif
