/*
 * inference-101
 */

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
    
    pixelOdd = (pixel % 2) ? true : false;
    lineOdd = ((pixel / width) % 2) ? true : false;
    
    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

//	printf("cuda thread %i %i  %i %i pixel %i \n", x, y, width, height, pixel);

#if 1
	// Convert to RGB
	if ((lineOdd) && (!pixelOdd))        
		dstImage[pixel] = make_float4(srcImage[pixel+width], srcImage[pixel], srcImage[pixel-1], 0.0f); // Green Info
	else if ((lineOdd) && (pixelOdd))   
		dstImage[pixel] = make_float4(srcImage[pixel+width+1], srcImage[pixel+1], srcImage[pixel], 0.0f); // Blue Info
#if 1
	if ((!lineOdd) && (!pixelOdd)) 
		dstImage[pixel] = make_float4(srcImage[pixel], srcImage[pixel-1], srcImage[pixel+width+1], 0.0f); // Red Info
	else if ((!lineOdd) && (pixelOdd)) 
		dstImage[pixel] = make_float4(srcImage[pixel+1], srcImage[pixel], srcImage[pixel+width], 0.0f); // Green Info
#endif

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


