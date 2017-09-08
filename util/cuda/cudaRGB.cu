/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaRGB.h"

//-------------------------------------------------------------------------------------------------------------------------

__global__ void RGBToRGBAf(uchar3* srcImage,
                           float4* dstImage,
                           uint32_t width,       uint32_t height)
{
    int x, y, pixel;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
    pixel = y * width + x;

    if (x >= width)
        return; 

    if (y >= height)
        return;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);
		
	const float  s  = 1.0f;
	const uchar3 px = srcImage[pixel];
	
	dstImage[pixel] = make_float4(px.x * s, px.y * s, px.z * s, 255.0f * s);
}

cudaError_t cudaRGBToRGBAf( uchar3* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(8,8,1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y), 1);

	RGBToRGBAf<<<gridDim, blockDim>>>( srcDev, destDev, width, height );
	
	return CUDA(cudaGetLastError());
}

