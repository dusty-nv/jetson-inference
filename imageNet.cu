/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "cudaUtility.h"



// gpuPreImageNet
__global__ void gpuPreImageNet( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z, px.y, px.x);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}


// cudaPreImageNet
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight,
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

	gpuPreImageNet<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight);

	return CUDA(cudaGetLastError());
}




// gpuPreImageNetMean
__global__ void gpuPreImageNetMean( float2 scale, float4* input, int iWidth, float* output, int oWidth, int oHeight, float3 mean_value )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int n = oWidth * oHeight;
	
	if( x >= oWidth || y >= oHeight )
		return;

	const int dx = ((float)x * scale.x);
	const int dy = ((float)y * scale.y);

	const float4 px  = input[ dy * iWidth + dx ];
	const float3 bgr = make_float3(px.z - mean_value.x, px.y - mean_value.y, px.x - mean_value.z);
	
	output[n * 0 + y * oWidth + x] = bgr.x;
	output[n * 1 + y * oWidth + x] = bgr.y;
	output[n * 2 + y * oWidth + x] = bgr.z;
}


// cudaPreImageNetMean
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight,
				             float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value )
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

	gpuPreImageNetMean<<<gridDim, blockDim>>>(scale, input, inputWidth, output, outputWidth, outputHeight, mean_value);

	return CUDA(cudaGetLastError());
}


