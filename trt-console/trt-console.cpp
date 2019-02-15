/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "tensorNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"


/**
 * Super Resolution Network
 */
class superResNet : public tensorNet
{
public:
	/**
	 * Load super resolution network
	 */
	static superResNet* Create();

	/**
	 * Destroy
	 */
	~superResNet();

	/**
	 * Upscale a 4-channel RGBA image.
	 */
	bool UpscaleRGBA( float* input, uint32_t inputWidth, uint32_t inputHeight,
			        float* output, uint32_t outputWidth, uint32_t outputHeight,
			        float maxPixelValue=255.0f );

	/**
	 * Upscale a 4-channel RGBA image.
	 */
	bool UpscaleRGBA( float* input, float* output, float maxPixelValue=255.0f );

	/**
	 * Retrieve the width of the input image, in pixels.
	 */
	inline uint32_t GetInputWidth() const						{ return mWidth; }

	/**
	 * Retrieve the height of the input image, in pixels.
	 */
	inline uint32_t GetInputHeight() const						{ return mHeight; }

	/**
	 * Retrieve the width of the output image, in pixels.
	 */
	inline uint32_t GetOutputWidth() const						{ return DIMS_W(mOutputs[0].dims); }

	/**
	 * Retrieve the height of the output image, in pixels.
	 */
	inline uint32_t GetOutputHeight() const						{ return DIMS_H(mOutputs[0].dims); }

	/**
	 * Retrieve the scale factor between the input and output.
	 */
	inline uint32_t GetScaleFactor() const						{ return GetOutputWidth() / GetInputWidth(); }

protected:
	superResNet()	{ }
};


// Create
superResNet* superResNet::Create()
{
	superResNet* net = new superResNet();

	const char* model_path  = "networks/Super-Resolution-BSD500/super_resolution_bsd500.onnx";
	const char* input_blob  = "input_0";
	const char* output_blob = "output_0";

	const uint32_t maxBatchSize = 1;

	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob, maxBatchSize) )
	{
		printf(LOG_TRT "failed to load superResNet model\n");
		return NULL;
	}

	printf("\n");
	printf("superResNet -- super resolution network loaded from:\n");
	printf("            -- model        '%s'\n", model_path);
	printf("            -- input blob   '%s'\n", input_blob);
	printf("            -- output blob  '%s'\n", output_blob);
	printf("            -- batch size   %u\n", maxBatchSize);
	printf("            -- input dims   %ux%u\n", net->GetInputWidth(), net->GetInputHeight());
	printf("            -- output dims  %ux%u\n", net->GetOutputWidth(), net->GetOutputHeight());
	printf("            -- scale factor %ux\n\n", net->GetScaleFactor());

	return net;
}


// Destructor
superResNet::~superResNet()
{

}

// cudaPreSuperResNet
cudaError_t cudaPreSuperResNet( float4* input, size_t inputWidth, size_t inputHeight,
				            float* output, size_t outputWidth, size_t outputHeight,
					       float maxPixelValue, cudaStream_t stream );

// cudaPostSuperResNet
cudaError_t cudaPostSuperResNet( float* input, size_t inputWidth, size_t inputHeight,
				             float4* output, size_t outputWidth, size_t outputHeight,
					        float maxPixelValue, cudaStream_t stream );

// UpscaleRGBA
bool superResNet::UpscaleRGBA( float* input, uint32_t inputWidth, uint32_t inputHeight,
		    				 float* output, uint32_t outputWidth, uint32_t outputHeight,
		    				 float maxPixelValue )
{
	/*
	 * convert input image to NCHW format and with pixel range 0.0-1.0f
	 */
	if( CUDA_FAILED(cudaPreSuperResNet((float4*)input, inputWidth, inputHeight,
								mInputCUDA, GetInputWidth(), GetInputHeight(), 
								maxPixelValue, GetStream())) )
	{
		printf(LOG_TRT "superResNet::UpscaleRGBA() -- cudaPreSuperResNet() failed\n");
		return false;
	}

	/*
	 * perform the inferencing
 	 */
	void* bindBuffers[] = { mInputCUDA, mOutputs[0].CUDA };	

	if( !mContext->execute(1, bindBuffers) )
	{
		printf(LOG_TRT "superResNet::UpscaleRGBA() -- failed to execute TensorRT network\n");
		return false;
	}

	PROFILER_REPORT();

	/*
	 * convert output image from NCHW to packed RGBA, with the user's pixel range
	 */
	if( CUDA_FAILED(cudaPostSuperResNet(mOutputs[0].CUDA, GetOutputWidth(), GetOutputHeight(),
								 (float4*)output, outputWidth, outputHeight, 
								 maxPixelValue, GetStream())) )
	{
		printf(LOG_TRT "superResNet::UpscaleRGBA() -- cudaPostSuperResNet() failed\n");
		return false;
	}

	return true;
}


// UpscaleRGBA
bool superResNet::UpscaleRGBA( float* input, float* output, float maxPixelValue )
{
	return UpscaleRGBA(input, GetInputWidth(), GetInputHeight(), output, GetOutputWidth(), GetOutputHeight(), maxPixelValue);
}


// print usage
int print_usage()
{
	printf("\nUSAGE:\n");
	printf("  trt-console --input=<path> --output=<path>\n\n");
	printf("     >  --input is a file path to the input image\n");
	printf("     >  --output is the path that the upscaled image will be written to\n");

     return 0;
}


// main entry point
int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	const char* inputPath = cmdLine.GetString("input");
	const char* outputPath = cmdLine.GetString("output");

	if( !inputPath || !outputPath )
	{
		printf("trt-console:  input and output image filenames required\n");
		return print_usage();
	}


	/*
	 * load super resolution network
	 */
	superResNet* net = superResNet::Create();

	if( !net )
	{
		printf("trt-console:  failed to load superResNet\n");
		return 0;
	}

	net->EnableProfiler();


	/* 
	 * load input image
	 */
	float* inputCPU = NULL;
	float* inputCUDA = NULL;

	int inputWidth = 0;
	int inputHeight = 0;

	if( !loadImageRGBA(inputPath, (float4**)&inputCPU, (float4**)&inputCUDA, &inputWidth, &inputHeight) )
	{
		printf("trt-console:  failed to load input image '%s'\n", inputPath);
		return 0;
	}


	/*
	 * allocate memory for output
	 */
	float* outputCPU = NULL;
	float* outputCUDA = NULL;

	const int outputWidth = inputWidth * net->GetScaleFactor();
	const int outputHeight = inputHeight * net->GetScaleFactor();

	if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputWidth * outputHeight * sizeof(float4)) )
	{
		printf("trt-console:  failed to allocate memory for %ix%i output image\n", outputWidth, outputHeight);
		return 0;
	}

	printf("trt-console:  input image size - %ix%i\n", inputWidth, inputHeight);
	printf("trt-console:  output image size - %ix%i\n", outputWidth, outputHeight);


	/*
	 * upscale image with network
	 */
	for( int i=0; i < 10; i++ )
	{
		if( !net->UpscaleRGBA(inputCUDA, inputWidth, inputHeight,
						  outputCUDA, outputWidth, outputHeight) )
		{
			printf("trt-console:  failed to process super resolution network\n");
			return 0;
		}
	}	


	/*
	 * save output image
	 */
	printf("trt-console:  saving %ix%i output image to '%s'\n", outputWidth, outputHeight, outputPath);

	if( !saveImageRGBA(outputPath, (float4*)outputCPU, outputWidth, outputHeight) )
	{
		printf("trt-console:  failed to save output image to '%s'\n", outputPath);
		return 0;
	}

	delete net;
	return 0;
}


