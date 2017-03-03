/*
 * inference-101
 */

#include <math_functions.h>
#include "cudaRGB.h"

//-------------------------------------------------------------------------------------------------------------------------

__global__ void RGBToRGBAf(uint8_t* srcImage,
                           float4* dstImage,
                           uint32_t width,       uint32_t height)
{
    int x, y, pixel;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
    pixel = y * width + x;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);

	const float s = 1;
	dstImage[pixel]     = make_float4(srcImage[pixel*3] * s, srcImage[ pixel*3 + 1] * s, srcImage[ pixel*3 + 2] * s, 0.0f);
}

cudaError_t cudaRGBToRGBAf( uint8_t* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(128,1,1);
	const dim3 gridDim(width/blockDim.x, height/blockDim.y, 1);

	RGBToRGBAf<<<gridDim, blockDim>>>( (uint8_t*)srcDev, destDev, width, height );

	return CUDA(cudaGetLastError());
}


//-------------------------------------------------------------------------------------------------------------------------

__global__ void BAYER_GR8toRGBA(uint8_t* srcImage,
                           float4* dstImage,
                           uint32_t width,       uint32_t height)
{
    int x, y, pixel;
	bool lineOdd, pixelOdd;

    x = (blockIdx.x * blockDim.x) + threadIdx.x;
    y = (blockIdx.y * blockDim.y) + threadIdx.y;
    pixel = y * width + x;
    
    pixelOdd = ((pixel) % 2) ? true : false;
	double t = floor((double)(pixel / width)) ;
    lineOdd = (int)t % 2 ? false : true;
    
    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);

#if 1 // Colour

/* BAYER_GR
 *    1  2  3  4  5  6 
 * 1  G  R  G  R  G  R
 * 2  B  G  B  G  B  G
 * 3  G  R  G  R  G  R
 * 4  B  G  B  G  B  G
 */
 
	// Odd lines
	if ((lineOdd) && (pixelOdd))  // First Pixel
	{
		int r = srcImage[pixel-1] + srcImage[pixel+1] /2;
		int b = srcImage[pixel+width]; // + srcImage[pixel-width+1] + srcImage[pixel-width-1] / 4;
		dstImage[pixel] = make_float4(r, srcImage[pixel], b, 0.0f); // Green Info
	}
	else if ((lineOdd) && (!pixelOdd))   
	{
		int g = srcImage[pixel-1] + srcImage[pixel+1] /2;
		int b = srcImage[pixel+width-1] + srcImage[pixel+width+1] / 2;
		dstImage[pixel] = make_float4(srcImage[pixel], g, b, 0.0f); // Red Info
	}

	// Even lines
	if ((!lineOdd) && (pixelOdd)) 
	{
		int g = srcImage[pixel+1] + srcImage[pixel-1] / 2;
		int r = srcImage[pixel+width-1] + srcImage[pixel+width+1] / 2;
		dstImage[pixel] = make_float4(r, g, srcImage[pixel], 0.0f); // Blue Info
	}
	else if ((!lineOdd) && (!pixelOdd)) 
	{
		int b = srcImage[pixel+1] + srcImage[pixel-1] / 2;
		int r = srcImage[pixel+width] + srcImage[pixel+width] / 2;
		dstImage[pixel] = make_float4(r, srcImage[pixel], b, 0.0f); // Green Info
	}

#else
	// Monochrome output
	dstImage[pixel]     = make_float4(srcImage[pixel], srcImage[ pixel], srcImage[ pixel], 0.0f);
#endif
}

cudaError_t cudaBAYER_GR8toRGBA( uint8_t* srcDev, float4* destDev, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	const dim3 blockDim(128,1,1);
	const dim3 gridDim(width/blockDim.x, height/blockDim.y, 1);

	BAYER_GR8toRGBA<<<gridDim, blockDim>>>( (uint8_t*)srcDev, destDev, width, height );

	return CUDA(cudaGetLastError());
}


