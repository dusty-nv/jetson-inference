/*
 * http://github.com/dusty-nv
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

