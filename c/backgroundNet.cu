/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "cudaFilterMode.cuh"
#include "cudaVector.h"

#include "imageFormat.h"


//#define RETAIN_ALPHA


// gpuBackgroundMask
template<typename T, cudaFilterMode filter, bool mask_alpha>
__global__ void gpuBackgroundMask( T* input, T* output, int width, int height, float* mask, int mask_width, int mask_height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int i = y * width + x;
	
	if( x >= width || y >= height )
		return;

	float mask_px = cudaFilterPixel<filter>(mask, x, y, mask_width, mask_height, width, height);
	
	if( mask_px < 0 ) mask_px = 0;
	if( mask_px > 1 ) mask_px = 1;
	
	if( !mask_alpha )
	{
		// retain the original alpha channel
		const float4 input_px = cast_vec<float4>(input[i]);
		
		const float4 output_px = make_float4(input_px.x * mask_px,
									  input_px.y * mask_px,
									  input_px.z * mask_px,
									  input_px.w);
		
		output[i] = cast_vec<T>(output_px);
	}
	else
	{
		// apply the mask to the alpha channel too
		output[i] = input[i] * mask_px;
	}
}


// cudaBackgroundMask
cudaError_t cudaBackgroundMask( void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
						  float* mask, uint32_t mask_width, uint32_t mask_height, bool mask_alpha,
						  cudaFilterMode filter, cudaStream_t stream )
{
	if( !input || !output || !mask )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0 || mask_width == 0 || mask_height == 0 )
		return cudaErrorInvalidValue;
		
	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	#define backgroundMaskFilter(type, maskAlpha, filterMode) \
		gpuBackgroundMask<type, maskAlpha, filterMode><<<gridDim, blockDim, 0, stream>>>( \
							(type*)input, (type*)output, width, height, \
							mask, mask_width, mask_height);

	#define backgroundMaskKernel(type) \
	{ \
		if( filter == FILTER_POINT && mask_alpha ) \
			backgroundMaskFilter(type, FILTER_POINT, true) \
		else if( filter == FILTER_POINT && !mask_alpha ) \
			backgroundMaskFilter(type, FILTER_POINT, false) \
		else if( filter == FILTER_LINEAR && mask_alpha ) \
			backgroundMaskFilter(type, FILTER_LINEAR, true) \
		else if( filter == FILTER_LINEAR && !mask_alpha ) \
			backgroundMaskFilter(type, FILTER_LINEAR, false) \
	}

	if( format == IMAGE_RGB8 )
		backgroundMaskKernel(uchar3)
	else if( format == IMAGE_RGBA8 )
		backgroundMaskKernel(uchar4)
	else if( format == IMAGE_RGB32F )
		backgroundMaskKernel(float3)
	else if( format == IMAGE_RGBA32F )
		backgroundMaskKernel(float4)
	else
	{
		imageFormatErrorMsg(LOG_CUDA, "cudaBackgroundMask()", format);
		return cudaErrorInvalidValue;
	}
		
	return CUDA(cudaGetLastError());
}


#if 0
	// note:  this seems unnecessary, because the mask min/max are almost always 0.0-1.0 anyways
	float mask_min = 1000000.0f;
	float mask_max = -1000000.0f;
	
	const uint32_t mask_width = GetOutputWidth();
	const uint32_t mask_height = GetOutputHeight();
	
	float* mask_ptr = mOutputs[0].CUDA;
	
	for( uint32_t y=0; y < mask_height; y++ )
	{
		for( uint32_t x=0; x < mask_width; x++ )
		{
			const float px = mask_ptr[y * mask_width + x];
			
			if( px < mask_min )
				mask_min = px;
			else if( px > mask_max )
				mask_max = px;
		}
	}
	
	printf("mask min=%f  max=%f\n", mask_min, mask_max);
	
	for( uint32_t y=0; y < mask_height; y++ )
		for( uint32_t x=0; x < mask_width; x++ )
			mask_ptr[y * mask_width + x] = (mask_ptr[y * mask_width + x] - mask_min) / (mask_max - mask_min);
#endif
