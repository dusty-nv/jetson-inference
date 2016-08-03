/*
 * inference-101
 */

#include "cudaResize.h"



// gpuResample
template <typename T>
__global__ void gpuResize( float2 scale, T* input, int iWidth, T* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const T px = input[ dy * iWidth + dx ];

	output[y*oWidth+x] = px;
}


// cudaResize
cudaError_t cudaResize( float* input, size_t inputWidth, size_t inputHeight,
				        float* output, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							          float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuResize<float><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}


// cudaResizeRGBA
cudaError_t cudaResizeRGBA( float4* input,  size_t inputWidth, size_t inputHeight,
				            float4* output, size_t outputWidth, size_t outputHeight )
{
	if( !input || !output )
		return cudaErrorInvalidDevicePointer;

	if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
		return cudaErrorInvalidValue;

	const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
							    float(inputHeight) / float(outputHeight) );

	// launch kernel
	const dim3 blockDim(8, 8);
	const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

	gpuResize<float4><<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}





