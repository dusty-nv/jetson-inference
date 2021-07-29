/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "depthNet.h"
#include "cudaUtility.h"

#include <float.h>


// reset the depth range to min/max
__global__ void gpuResetDepthRange( int2* range )
{
	range[0] = make_int2(INT_MAX, INT_MIN);
}

// calculate the range of depths
__global__ void gpuDepthRange( float* data, int width, int height, int2* range )
{
	__shared__ int2 range_shared;
	
	if( threadIdx.x == 0 && threadIdx.y == 0 )
		range_shared = make_int2(INT_MAX, INT_MIN);
	
	__syncthreads();
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x < width && y < height )
	{
		const int px = data[y * width + x] * DEPTH_FLOAT_TO_INT;
		
		atomicMin(&range_shared.x, px);
		atomicMax(&range_shared.y, px);
	}
	
	__syncthreads();
	
	if( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		atomicMin(&range->x, range_shared.x);
		atomicMax(&range->y, range_shared.y);
	}
}
	
// compute histogram
__global__ void gpuDepthHistogram( float* data, int width, int height, int2* range, unsigned int* histogram )
{
	__shared__ unsigned int hist_shared[DEPTH_HISTOGRAM_BINS];
	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	const int inner_idx = threadIdx.y * blockDim.x + threadIdx.x;

	// initialize shared memory
	if( threadIdx.y < (DEPTH_HISTOGRAM_BINS / blockDim.x) )
		hist_shared[inner_idx] = 0;

	__syncthreads();
  
	// each thread block merges the value in its own shared memory into global memory
	if( x < width && y < height )
	{
		const int2 range_int = range[0];
		const float2 range_float = make_float2((float)range_int.x / DEPTH_FLOAT_TO_INT,
									    (float)range_int.y / DEPTH_FLOAT_TO_INT);
									    
		const float p = data[y * width + x];
		const float r = (p - range_float.x) / (range_float.y - range_float.x) * 255.0f;
		
		atomicAdd(&(hist_shared[(int)r]), 1);
	}
	
	__syncthreads();

	// accumulate in global memory
	if( threadIdx.y < (DEPTH_HISTOGRAM_BINS / blockDim.x) )
		atomicAdd(&(histogram[inner_idx]), hist_shared[inner_idx]);
}
 
// normalize histogram to probabilities
__global__ void gpuDepthHistogramPDF( unsigned int* histogram, float* histogramPDF, int size )
{
    histogramPDF[threadIdx.x] = (float)histogram[threadIdx.x] / size;
}

// cumulative histogram probabilities 
__global__ void gpuDepthHistogramCDF( float* histogramPDF, float* histogramCDF )
{
#if 1
	__shared__ float pdf_shared[DEPTH_HISTOGRAM_BINS];
	__shared__ float cdf_shared[DEPTH_HISTOGRAM_BINS];

	pdf_shared[threadIdx.x] = histogramPDF[threadIdx.x];
	cdf_shared[threadIdx.x] = 0;
	
	__syncthreads();
	
	if( threadIdx.x == 0 )
	{
		cdf_shared[0] = pdf_shared[0];
		
		for( int n=1; n < DEPTH_HISTOGRAM_BINS; n++ )
			cdf_shared[n] = cdf_shared[n-1] + pdf_shared[n];
	}
	
	__syncthreads();
	
	histogramCDF[threadIdx.x] = cdf_shared[threadIdx.x];
	
#else
	__shared__ float sum_shared[DEPTH_HISTOGRAM_BINS];

	const int id = threadIdx.x + blockIdx.x * blockDim.x;
	const int tid = threadIdx.x;
	
	// transfer global memory data to shared memory
	sum_shared[tid] = histogramPDF[id];
	__syncthreads(); 

	for( int stride = blockDim.x/2; stride > 0; stride /= 2 )
	{
		if( tid < stride )
			sum_shared[tid] += sum_shared[tid + stride];

		__syncthreads();
	} 

	if( tid == 0 )
		histogramCDF[blockIdx.x] = sum_shared[0];
#endif
}

// equalize histogram
__global__ void gpuDepthHistogramEDU( float* histogramCDF, unsigned int* histogramEDU )
{
	histogramEDU[threadIdx.x] = (unsigned int)(255.0f * histogramCDF[threadIdx.x] + 0.5f);
}

// apply histogram to image
__global__ void gpuDepthHistogramMap( unsigned int* histogramEDU, float* depth_in, float* depth_out, int width, int height, int2* range )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if( x >= width || y >= height )
		return;
	
	const int2 range_int = range[0];
	const float2 range_float = make_float2((float)range_int.x / DEPTH_FLOAT_TO_INT,
								    (float)range_int.y / DEPTH_FLOAT_TO_INT);
								    
	const float p = depth_in[y * width + x];
	const float r = (p - range_float.x) / (range_float.y - range_float.x) * 255.0f;
		
	depth_out[y * width + x] = histogramEDU[(int)r];
}


// histogramEqualization (CUDA)
bool depthNet::histogramEqualizationCUDA()
{
	float* depth_field = GetDepthField();
	
	const int depth_width = GetDepthFieldWidth();
	const int depth_height = GetDepthFieldHeight();
	const int depth_size = depth_width * depth_height;
	
	CUDA(cudaMemsetAsync(mHistogram, 0, DEPTH_HISTOGRAM_BINS * sizeof(uint32_t), GetStream()));
	
	const dim3 blockDim(16, 16);
	const dim3 gridDim(iDivUp(GetDepthFieldWidth(), blockDim.x), iDivUp(GetDepthFieldHeight(), blockDim.y));
	
	// get the range of raw depth values
	gpuResetDepthRange<<<1, 1, 0, GetStream()>>>(mDepthRange);
	gpuDepthRange<<<gridDim, blockDim, 0, GetStream()>>>(depth_field, depth_width, depth_height, mDepthRange);
	
	// compute the histogram
	gpuDepthHistogram<<<gridDim, blockDim, 0, GetStream()>>>(depth_field, depth_width, depth_height, mDepthRange, mHistogram);
	
	// equalize the histogram
	gpuDepthHistogramPDF<<<1, DEPTH_HISTOGRAM_BINS, 0, GetStream()>>>(mHistogram, mHistogramPDF, depth_size);
	gpuDepthHistogramCDF<<<1, DEPTH_HISTOGRAM_BINS, 0, GetStream()>>>(mHistogramPDF, mHistogramCDF);
	gpuDepthHistogramEDU<<<1, DEPTH_HISTOGRAM_BINS, 0, GetStream()>>>(mHistogramCDF, mHistogramEDU);
	
	// re-map the depth image
	gpuDepthHistogramMap<<<gridDim, blockDim, 0, GetStream()>>>(mHistogramEDU, depth_field, mDepthEqualized, depth_width, depth_height, mDepthRange);
	
	if( CUDA_FAILED(cudaGetLastError()) )
		return false;
	
	return true;
}


// histogramEqualization (CPU)
bool depthNet::histogramEqualization()
{
	float* depth_field = GetDepthField();
	
	const uint32_t depth_width = GetDepthFieldWidth();
	const uint32_t depth_height = GetDepthFieldHeight();
	const uint32_t depth_size = depth_width * depth_height;
	
	// get range of raw depth data
	float2 depthRange = make_float2(FLT_MAX, FLT_MIN);

	for( uint32_t y=0; y < depth_height; y++ )
	{
		for( uint32_t x=0; x < depth_width; x++ )
		{
			const float depth = depth_field[y * depth_width + x];

			if( depth < depthRange.x )
				depthRange.x = depth;

			if( depth > depthRange.y )
				depthRange.y = depth;
		}
	}
	
	const float depthSpan = depthRange.y - depthRange.x;
	LogVerbose("depthNet -- depth range:  %f -> %f\n", depthRange.x, depthRange.y);
	
	// rescale to [0,255]
	for( uint32_t y=0; y < depth_height; y++ )
	{
		for( uint32_t x=0; x < depth_width; x++ )
		{
			float px = depth_field[y * depth_width + x];
			
			if( px > depthRange.y ) px = depthRange.y;
			if( px < depthRange.x ) px = depthRange.x;
			
			mDepthEqualized[y * depth_width + x] = ((px - depthRange.x) / depthSpan) * 255.0f;
		}
	}

	// histogram
	uint32_t hist[DEPTH_HISTOGRAM_BINS] = {0};
	
	for( uint32_t y=0; y < depth_height; y++ )
		for( uint32_t x=0; x < depth_width; x++ )
			hist[(int)mDepthEqualized[y * depth_width + x]]++;

	// histogram probability
	float histPDF[DEPTH_HISTOGRAM_BINS] = {0};
	
	for( uint32_t n=0; n < DEPTH_HISTOGRAM_BINS; n++ )
		histPDF[n] = (float)hist[n] / depth_size;

	// cumulative histogram
	float histCDF[DEPTH_HISTOGRAM_BINS] = {0};
	
	histCDF[0] = histPDF[0];
	
	for( uint32_t n=1; n < DEPTH_HISTOGRAM_BINS; n++ )
		histCDF[n] = histCDF[n-1] + histPDF[n];
		
	// histogram equalization
	int histEDU[DEPTH_HISTOGRAM_BINS] = {0};
	
	for( uint32_t n=0; n < DEPTH_HISTOGRAM_BINS; n++ )
		histEDU[n] = (int)(255.0f * histCDF[n] + 0.5f);
	
	// histogram mapping
	for( uint32_t y=0; y < depth_height; y++ )
		for( uint32_t x=0; x < depth_width; x++ )
			mDepthEqualized[y * depth_width + x] = histEDU[(int)mDepthEqualized[y * depth_width + x]];
		
	return true;
}


