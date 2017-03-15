#include "camera.h"
#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"


bool camera::ConvertBAYER_GR8toRGBA( void* input, void** output )
{	
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "gvCamera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}
	
	// USB webcam is RGB
	if( CUDA_FAILED(cudaBAYER_GR8toRGBA((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
	{
		printf(LOG_CUDA "gvCamera -- conversion cudaRGBToRGBAf failed\n");
		return false;
	}

	*output = mRGBA;
	return true;
}

bool camera::ConvertYUVtoRGBf( void* input, void** output )
{
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "cudaMalloc -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}

	// nvcamera is YUV
	if( CUDA_FAILED(cudaYUVToRGBAf((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
		{
			printf(LOG_CUDA "cudaYUVToRGBAf -- failed convert %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}

	*output = mRGBA;

	return true;
}

// ConvertRGBA
bool camera::ConvertNV12toRGBA( void* input, void** output )
{	
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "gstCamera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}
	
	// nvcamera is NV12
	if( CUDA_FAILED(cudaNV12ToRGBAf((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
		return false;
	
	*output = mRGBA;
	return true;
}

bool camera::ConvertYUVtoRGBA( void* input, void** output )
{	
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "gstCamera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}
	
	// nvcamera is YUV
	if( CUDA_FAILED(cudaYUVToRGBAf((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
		return false;
	
	*output = mRGBA;
	return true;
}

bool camera::ConvertRGBtoRGBA( void* input, void** output )
{	
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "camera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}
	
	// USB webcam is RGB
	if( CUDA_FAILED(cudaRGBToRGBAf((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
	{
		printf(LOG_CUDA "camera -- conversion cudaRGBToRGBAf failed (%dx%d)\n", mWidth, mHeight);
		return false;
	}

	*output = mRGBA;
	return true;
}

