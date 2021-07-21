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

	const int px_idx = y * width + x;
	T px = input[px_idx];
	
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

			px.x = alpha * color.x + ialph * px.x;
			px.y = alpha * color.y + ialph * px.y;
			px.z = alpha * color.z + ialph * px.z;
		}
	}
	
	output[px_idx] = px;	 
}


template<typename T>
__global__ void gpuDetectionOverlayBox( T* input, T* output, int imgWidth, int imgHeight, int x0, int y0, int boxWidth, int boxHeight, const float4 color ) 
{
	const int box_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int box_y = blockIdx.y * blockDim.y + threadIdx.y;

	if( box_x >= boxWidth || box_y >= boxHeight )
		return;

	const int x = box_x + x0;
	const int y = box_y + y0;

	if( x >= imgWidth || y >= imgHeight )
		return;

	T px = input[ y * imgWidth + x ];

	const float alpha = color.w / 255.0f;
	const float ialph = 1.0f - alpha;

	px.x = alpha * color.x + ialph * px.x;
	px.y = alpha * color.y + ialph * px.y;
	px.z = alpha * color.z + ialph * px.z;
	
	output[y * imgWidth + x] = px;
}

template<typename T>
cudaError_t launchDetectionOverlay( T* input, T* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors )
{
	if( !input || !output || width == 0 || height == 0 || !detections || numDetections == 0 || !colors )
		return cudaErrorInvalidValue;
			
	// this assumes that the output already has the input image copied to it,
	// which if input != output, is done first by detectNet::Detect()
	for( int n=0; n < numDetections; n++ )
	{
		const int boxWidth = (int)detections[n].Width();
		const int boxHeight = (int)detections[n].Height();

		// launch kernel
		const dim3 blockDim(8, 8);
		const dim3 gridDim(iDivUp(boxWidth,blockDim.x), iDivUp(boxHeight,blockDim.y));

		gpuDetectionOverlayBox<T><<<gridDim, blockDim>>>(input, output, width, height, (int)detections[n].Left, (int)detections[n].Top, boxWidth, boxHeight, colors[detections[n].ClassID]); 
	}

	return cudaGetLastError();
}

cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections, float4* colors )
{
	if( format == IMAGE_RGB8 )
		return launchDetectionOverlay<uchar3>((uchar3*)input, (uchar3*)output, width, height, detections, numDetections, colors); 
	else if( format == IMAGE_RGBA8 )
		return launchDetectionOverlay<uchar4>((uchar4*)input, (uchar4*)output, width, height, detections, numDetections, colors);  
	else if( format == IMAGE_RGB32F )
		return launchDetectionOverlay<float3>((float3*)input, (float3*)output, width, height, detections, numDetections, colors);  
	else if( format == IMAGE_RGBA32F )
		return launchDetectionOverlay<float4>((float4*)input, (float4*)output, width, height, detections, numDetections, colors); 
	else
		return cudaErrorInvalidValue;
}

