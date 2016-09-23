/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaOverlay.h"


static inline __device__ __host__ bool eq_less( float a, float b, float epsilon )
{
	return (a > (b - epsilon) && a < (b + epsilon)) ? true : false;
}

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
	
	const float fx = x;
	const float fy = y;
	
	const float thick = 10.0f;
	const float alpha = color.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	for( int nr=0; nr < numRects; nr++ )
	{
		const float4 r = rects[nr];
		
		//printf("%i %i %i  %f %f %f %f\n", numRects, x, y, r.x, r.y, r.z, r.w);
		
		if( fy >= r.y && fy <= r.w /*&& (eq_less(fx, r.x, ep) || eq_less(fx, r.z, ep))*/ )
		{
			if( fx >= r.x && fx <= r.z /*&& (eq_less(fy, r.y, ep) || eq_less(fy, r.w, ep))*/ )
			{
				//printf("cuda rect %i %i\n", x, y);

				px_out.x = alpha * color.x + ialph * px_out.x;
				px_out.y = alpha * color.y + ialph * px_out.y;
				px_out.z = alpha * color.z + ialph * px_out.z;
			}
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
