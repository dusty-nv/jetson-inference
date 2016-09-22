/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaOverlay.h"



template<typename T>
__global__ void gpuRectOutlines( T* input, T* output, int width, int height,
						        float4* rects, int numRects, float4 color ) 
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px_in = input[ y * width + x ];
	T px_out = px_in;
	
	for( int nr=0; nr < numRects; nr++ )
	{
		const float4 r = rects[nr];
		
		if( (x >= r.x && x <= r.z && (y == r.y || y == r.w)) ||
			(y >= r.y && y <= r.w && (x == r.x || x == r.z)) )
		{
			px_out = color;
		}
	}
	
	output[y * width + x] = px_out;	 
}


cudaError_t cudaRectOutlineOverlay( float4* input, float4* output, uint32_t width, uint32_t height, float4* boundingBoxes, int numBoxes, const float4& color )
{
	if( !input || !output || width == 0 || height == 0 || !boundingBoxes || numBoxes == 0 )
		return cudaErrorInvalidValue;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuRectOutlines<float4><<<gridDim, blockDim>>>(input, output, width, height, boundingBoxes, numBoxes, color); 

	return cudaGetLastError();
}
