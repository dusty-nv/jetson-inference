/*
 * inference-101
 */

#include "cudaNormalize.h"



// gpuNormalize
template <typename T>
__global__ void gpuNormalize( T* input, T* output, int width, int height, float scaling_factor )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= width || y >= height )
		return;

	const T px = input[ y * width + x ];

	output[y*width+x] = make_float4(px.x * scaling_factor,
							  px.y * scaling_factor,
							  px.z * scaling_factor,
							  px.w * scaling_factor);
}


// cudaNormalizeRGBA
cudaError_t cudaNormalizeRGBA( float4* input, const float2& input_range,
						 float4* output, const float2& output_range,
						 size_t  width,  size_t height )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( width == 0 || height == 0  )
		return cudaErrorInvalidValue;

	const float multiplier = output_range.y / input_range.y;

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height,blockDim.y));

	gpuNormalize<float4><<<gridDim, blockDim>>>(input, output, width, height, multiplier);

	return CUDA(cudaGetLastError());
}





