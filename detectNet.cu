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

#include "detectNet.h"
#include "cudaUtility.h"



template<typename T>
__global__ void gpuDetectionOverlay( T* input, T* output, int width, int height, detectNet::Detection* detections, int numDetections, float4* colors ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px_in = input[ y * width + x ];
	T px_out = px_in;
	
	const float fx = x;
	const float fy = y;
	
	for( int n=0; n < numDetections; n++ )
	{
		const detectNet::Detection det = detections[n];

		// check if this pixel is inside the bounding box
		if( fx >= det.Left && fx <= det.Right && fy >= det.Top && fy <= det.Bottom )
		{
			const float4 color = colors[det.ClassID];	

			const float alpha = color.w / 255.0f;
			const float ialph = 1.0f - alpha;

			px_out.x = alpha * color.x + ialph * px_out.x;
			px_out.y = alpha * color.y + ialph * px_out.y;
			px_out.z = alpha * color.z + ialph * px_out.z;
		}
	}
	
	output[y * width + x] = px_out;	 
}

cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors )
{
	if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0 || !colors )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuDetectionOverlay<float4><<<gridDim, blockDim>>>(input, output, width, height, detections, numDetections, colors); 

	return cudaGetLastError();
}

