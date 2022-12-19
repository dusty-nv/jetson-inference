/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "backgroundNet.h"
#include "tensorConvert.h"
#include "modelDownloader.h"

#include "commandLine.h"
#include "filesystem.h"


// constructor
backgroundNet::backgroundNet() : tensorNet()
{

}


// destructor
backgroundNet::~backgroundNet()
{

}


// Create
backgroundNet* backgroundNet::Create( const char* network, uint32_t maxBatchSize, 
					             precisionType precision, deviceType device, bool allowGPUFallback )
{
	nlohmann::json model;
	
	if( !DownloadModel(BACKGROUNDNET_MODEL_TYPE, network, model) )
		return NULL;
	
	std::string model_dir = "networks/" + model["dir"].get<std::string>() + "/";
	std::string model_path = model_dir + JSON_STR(model["model"]);;
	std::string input = JSON_STR_DEFAULT(model["input"], BACKGROUNDNET_DEFAULT_INPUT);
	std::string output = JSON_STR_DEFAULT(model["output"], BACKGROUNDNET_DEFAULT_OUTPUT);
		
	return Create(model_path.c_str(), input.c_str(), output.c_str(), 
			    maxBatchSize, precision, device, allowGPUFallback);
}


// Create
backgroundNet* backgroundNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
backgroundNet* backgroundNet::Create( const commandLine& cmdLine )
{
	backgroundNet* net = NULL;

	// obtain the network name
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "u2net");
	
	// parse the network type
	if( !FindModel(BACKGROUNDNET_MODEL_TYPE, modelName) )
	{
		const char* input  = cmdLine.GetString("input_blob", BACKGROUNDNET_DEFAULT_INPUT);
		const char* output = cmdLine.GetString("output_blob", BACKGROUNDNET_DEFAULT_OUTPUT);

		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = backgroundNet::Create(modelName, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = backgroundNet::Create(modelName, DEFAULT_MAX_BATCH_SIZE);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// Create
backgroundNet* backgroundNet::Create( const char* model_path, const char* input, const char* output, 
						        uint32_t maxBatchSize, precisionType precision, 
						        deviceType device, bool allowGPUFallback )
{
	// check for built-in model string
	if( FindModel(BACKGROUNDNET_MODEL_TYPE, model_path) )
	{
		return Create(model_path, maxBatchSize, precision, device, allowGPUFallback);
	}
	else if( fileExtension(model_path).length() == 0 )
	{
		LogError(LOG_TRT "couldn't find built-in background model '%s'\n", model_path);
		return NULL;
	}
	
	// load custom model
	backgroundNet* net = new backgroundNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(model_path, input, output, maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;
	
	return net;
}


// init
bool backgroundNet::init( const char* model_path, const char* input, const char* output, 
				      uint32_t maxBatchSize, precisionType precision, 
				      deviceType device, bool allowGPUFallback )
{
	if( !model_path || !input || !output )
		return NULL;
	
	LogInfo("\n");
	LogInfo("backgroundNet -- loading background network from:\n");
	LogInfo("           -- model       %s\n", model_path);
	LogInfo("           -- input_blob  '%s'\n", input);
	LogInfo("           -- output_blob '%s'\n", output);
	LogInfo("           -- batch_size  %u\n\n", maxBatchSize);
	
	// load model
	if( !tensorNet::LoadNetwork(NULL, model_path, NULL, input, output, 
					        maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "failed to load %s\n", model_path);
		return false;
	}
		
	return true;
}
	

// from backgroundNet.cu
cudaError_t cudaBackgroundMask( void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
						  float* mask, uint32_t mask_width, uint32_t mask_height, bool mask_alpha,
						  cudaFilterMode filter, cudaStream_t stream );
						  
						  
// Process
bool backgroundNet::Process( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, cudaFilterMode filter, bool maskAlpha )
{
	// verify parameters
	if( !input || !output || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "backgroundNet::Process() called with NULL / invalid parameters\n");
		return false;
	}

	// preprocess image
	PROFILER_BEGIN(PROFILER_PREPROCESS);
	
	if( CUDA_FAILED(cudaTensorNormMeanRGB(input, format, width, height,
								   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								   make_float2(0.0f, 1.0f),
								   make_float3(0.485f, 0.456f, 0.406f),
								   make_float3(0.229f, 0.224f, 0.225f),
								   GetStream())) )
	{
		return false;
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	
	// process with TRT
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);
	
	// apply the background mask
	if( CUDA_FAILED(cudaBackgroundMask(input, output, width, height, format,
								mOutputs[0].CUDA, GetOutputWidth(), GetOutputHeight(),
								maskAlpha, filter, GetStream())) )
	{
		return false;
	}
		
	PROFILER_END(PROFILER_POSTPROCESS);

	return true;
}
	