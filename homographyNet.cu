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


// rgbaToGray
__device__ inline float rgbaToGray( const float4& rgba )
{
	return rgba.x * 0.2989f + rgba.y * 0.5870f + rgba.z * 0.1140f;
}


// normalize to [-1,1]
__device__ inline float norm1( float value )
{
	return value / 255.0f * 2.0f - 1.0f;
}


// gpuPreHomographyNet
__global__ void gpuPreHomographyNet( float2 scale, float4* in_A, float4* in_B, int in_width, 
							  float* output, int out_width, int out_height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = out_width * out_height;
	
	if( x >= out_width || y >= out_height )
		return;

	// scale coordinates to input
	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	// convert inputs to grayscale
	const int in_idx = dy * in_width + dx;

	const float4 rgba_A = in_A[in_idx];
	const float4 rgba_B = in_B[in_idx];

	const float gray_A = rgbaToGray(rgba_A);
	const float gray_B = rgbaToGray(rgba_B);

	const float norm_A = norm1(gray_A);
	const float norm_B = norm1(gray_B);

	// concatenate the images
	output[n * 0 + y * out_width + x] = norm_A;
	output[n * 1 + y * out_width + x] = norm_B;
}


// cudaPreHomographyNet
cudaError_t cudaPreHomographyNet( float4* inputA, float4* inputB, size_t inputWidth, size_t inputHeight,
				         	    float* output, size_t outputWidth, size_t outputHeight,
					         cudaStream_t stream )
{
	if( !inputA || !inputB || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuPreHomographyNet<<<gridDim, blockDim, 0, stream>>>(scale, inputA, inputB, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}

