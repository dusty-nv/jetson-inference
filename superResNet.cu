/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#include "cudaUtility.h"



// clip float to [min,max]
static inline __device__ float clip( const float x, float min, float max )
{
	return x > max ? max : x < min ? min : x;
}


// clip vector to [min,max]
static inline __device__ float4 clip( const float4& px, float min, float max )
{
	return make_float4(clip(px.x, min, max),
				    clip(px.y, min, max),
				    clip(px.z, min, max),
				    clip(px.w, min, max));
}


// gpuPreSuperResNet
template<typename T>
__global__ void gpuPreSuperResNet( T* input, int iWidth, float* output, int oWidth, int oHeight, float2 res_scale, float pixel_scale )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * res_scale.x);
	const int dy = ((float)y * res_scale.y);

	const T px = input[ dy * iWidth + dx ];
	const float3 rgb = make_float3(px.x * pixel_scale, px.y * pixel_scale, px.z * pixel_scale);
	
	output[n * 0 + y * oWidth + x] = rgb.x;
	output[n * 1 + y * oWidth + x] = rgb.y;
	output[n * 2 + y * oWidth + x] = rgb.z;
}


// cudaPreSuperResNet
cudaError_t cudaPreSuperResNet( float4* input, size_t inputWidth, size_t inputHeight,
				            float* output, size_t outputWidth, size_t outputHeight,
					       float maxPixelValue, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 res_scale = make_float2( float(inputWidth) / float(outputWidth),
							        float(inputHeight) / float(outputHeight) );

	const float pixel_scale = 1.0f / maxPixelValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreSuperResNet<float4><<<gridDim, blockDim, 0, stream>>>(input, inputWidth, output, outputWidth, outputHeight, res_scale, pixel_scale);

	return CUDA(cudaGetLastError());
}


// gpuPostSuperResNet
template<typename T>
__global__ void gpuPostSuperResNet( float* input, int iWidth, int iHeight, T* output, int oWidth, int oHeight, float2 res_scale, float pixel_scale )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = iWidth * iHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * res_scale.x);
	const int dy = ((float)y * res_scale.y);

	const float4 rgb = clip(make_float4(input[n * 0 + dy * iWidth + dx] * pixel_scale,
							      input[n * 1 + dy * iWidth + dx] * pixel_scale,
							      input[n * 2 + dy * iWidth + dx] * pixel_scale,
							      pixel_scale), 0.0f, pixel_scale);

	output[y * oWidth + x] = rgb;
}


// cudaPostSuperResNet
cudaError_t cudaPostSuperResNet( float* input, size_t inputWidth, size_t inputHeight,
				             float4* output, size_t outputWidth, size_t outputHeight,
					        float maxPixelValue, cudaStream_t stream )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 res_scale = make_float2( float(inputWidth) / float(outputWidth),
							        float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPostSuperResNet<float4><<<gridDim, blockDim, 0, stream>>>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, res_scale, maxPixelValue);

	return CUDA(cudaGetLastError());
}

