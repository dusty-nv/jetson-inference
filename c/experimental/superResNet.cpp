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

#include "superResNet.h"
#include "cudaUtility.h"


// constructor
superResNet::superResNet()
{

}


// Destructor
superResNet::~superResNet()
{

}


// Create
superResNet* superResNet::Create()
{
#ifndef HAS_SUPERRES_NET
	printf(LOG_TRT "error -- superResNet is supported only in TensorRT 5.0 and newer\n");
	return NULL;
#endif

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


// cudaPreSuperResNet (from superResNet.cu)
cudaError_t cudaPreSuperResNet( float4* input, size_t inputWidth, size_t inputHeight,
				            float* output, size_t outputWidth, size_t outputHeight,
					       float maxPixelValue, cudaStream_t stream );

// cudaPostSuperResNet (from superResNet.cu)
cudaError_t cudaPostSuperResNet( float* input, size_t inputWidth, size_t inputHeight,
				             float4* output, size_t outputWidth, size_t outputHeight,
					        float maxPixelValue, cudaStream_t stream );

// UpscaleRGBA
bool superResNet::UpscaleRGBA( float* input, uint32_t inputWidth, uint32_t inputHeight,
		    				 float* output, uint32_t outputWidth, uint32_t outputHeight,
		    				 float maxPixelValue )
{
	PROFILER_BEGIN(PROFILER_PREPROCESS);

	/*
	 * convert input image to NCHW format and with pixel range 0.0-1.0f
	 */
	if( CUDA_FAILED(cudaPreSuperResNet((float4*)input, inputWidth, inputHeight,
								mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
								maxPixelValue, GetStream())) )
	{
		printf(LOG_TRT "superResNet::UpscaleRGBA() -- cudaPreSuperResNet() failed\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	/*
	 * perform the inferencing
 	 */
	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

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

	PROFILER_END(PROFILER_POSTPROCESS);
	return true;
}


// UpscaleRGBA
bool superResNet::UpscaleRGBA( float* input, float* output, float maxPixelValue )
{
	return UpscaleRGBA(input, GetInputWidth(), GetInputHeight(), output, GetOutputWidth(), GetOutputHeight(), maxPixelValue);
}

