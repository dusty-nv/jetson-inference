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
 
#include "depthNet.h"
#include "imageNet.cuh"

#include "commandLine.h"


// constructor
depthNet::depthNet() : tensorNet()
{
	mNetworkType = CUSTOM;
}


// destructor
depthNet::~depthNet()
{

}


// NetworkTypeFromStr
depthNet::NetworkType depthNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return depthNet::CUSTOM;

	depthNet::NetworkType type = depthNet::MOBILENET;

	// ONNX models
	if( strcasecmp(modelName, "mobilenet") == 0 )
		type = depthNet::MOBILENET;
	else if( strcasecmp(modelName, "resnet-18") == 0 || strcasecmp(modelName, "resnet_18") == 0 || strcasecmp(modelName, "resnet18") == 0 )
		type = depthNet::RESNET_18;
	else if( strcasecmp(modelName, "resnet-50") == 0 || strcasecmp(modelName, "resnet_50") == 0 || strcasecmp(modelName, "resnet50") == 0 )
		type = depthNet::RESNET_50;
	else
		type = depthNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* depthNet::NetworkTypeToStr( depthNet::NetworkType type )
{
	switch(type)
	{
		case MOBILENET:	return "MonoDepth-Mobilenet";
		case RESNET_18:	return "MonoDepth-ResNet18";
		case RESNET_50:	return "MonoDepth-ResNet50";
		default:			return "Custom";
	}
}


// Create
depthNet* depthNet::Create( depthNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	depthNet* net = NULL;
	
	if( networkType == MOBILENET )
		net = Create("networks/DepthNet-Mobilenet/depth_mobilenet.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == RESNET_18 )
		net = Create("networks/DepthNet-ResNet18/depth_resnet18.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == RESNET_50 )
		net = Create("networks/DepthNet-ResNet50/depth_resnet50.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);

	if( net != NULL )
		net->mNetworkType = networkType;

	return net;
}


// Create
depthNet* depthNet::Create( const char* model_path, const char* input_blob, const char* output_blob,
					   uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	depthNet* net = new depthNet();
	
	if( !net )
		return NULL;
	
	printf("\n");
	printf("depthNet -- loading mono depth network model from:\n");
	printf("         -- model:      %s\n", model_path);
	printf("         -- input_blob  '%s'\n", input_blob);
	printf("         -- output_blob '%s'\n", output_blob);
	printf("         -- batch_size  %u\n\n", maxBatchSize);
	
	// load network
	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("depthNet -- failed to initialize.\n");
		return NULL;
	}

	// load colormaps
	CUDA(cudaColormapInit());

	// return network
	return net;
}


// Create
depthNet* depthNet::Create( int argc, char** argv )
{
	depthNet* net = NULL;

	// obtain the network name
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "mobilenet");
	
	// parse the network type
	const depthNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == depthNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input  = DEPTHNET_DEFAULT_INPUT;
		if( !output )  output = DEPTHNET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = depthNet::Create(modelName, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = depthNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// Process
bool depthNet::Process( float* input, uint32_t input_width, uint32_t input_height,
				    float* output, uint32_t output_width, uint32_t output_height, 
                        cudaColormapType colormap, cudaFilterMode filter )
{
	if( !input || !output || input_width == 0 || input_height == 0 || output_width == 0 || output_height == 0 )
	{
		printf("depthNet::Process( 0x%p, 0x%p, %u, %u ) -> invalid parameters\n", input, output, input_width, input_height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_ONNX) )
	{
		// remap from [0,255] -> [0,1], no mean pixel subtraction or std dev applied
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)input, input_width, input_height, 
										   mInputCUDA, mWidth, mHeight, 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.0f, 0.0f, 0.0f),
										   make_float3(1.0f, 1.0f, 1.0f), 
										   GetStream())) )
		{
			printf(LOG_TRT "depthNet::Process() -- cudaPreImageNetNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		printf("depthNet::Process() -- support for models other than ONNX not implemented.\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "depthNet::Process() -- failed to execute TensorRT context\n");
		return false;
	}

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	const int depth_width = DIMS_W(mOutputs[0].dims);
	const int depth_height = DIMS_H(mOutputs[0].dims);
	const int depth_channels = DIMS_C(mOutputs[0].dims);

	// find the min/max depth range
	float2 depthRange = make_float2(100000000.0f, -100000000.0f);

	for( int y=0; y < depth_height; y++ )
	{
		for( int x=0; x < depth_height; x++ )
		{
			const float depth = mOutputs[0].CPU[y * depth_width + x];

			if( depth < depthRange.x )
				depthRange.x = depth;

			if( depth > depthRange.y )
				depthRange.y = depth;
		}
	}

	//printf("depth image:  %i x %i x %i\n", depth_width, depth_height, depth_channels);
	printf("depth range:  %f -> %f\n", depthRange.x, depthRange.y);

	//depthRange = make_float2(0.95f, 5.0f);

	PROFILER_END(PROFILER_POSTPROCESS);
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// apply color mapping to depth image
	if( CUDA_FAILED(cudaColormap(mOutputs[0].CUDA, depth_width, depth_height,
						    (float4*)output, output_width, output_height,
						    depthRange, colormap, filter, GetStream())) )
	{
		printf("depthNet::Process() -- failed to map depth image with cudaColormap()\n");
		return false; 
	}

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}

// Process
bool depthNet::Process( float* input, float* output, uint32_t width, uint32_t height, cudaColormapType colormap, cudaFilterMode filter )
{
	return Process(input, width, height, output, width, height, colormap, filter);
}


