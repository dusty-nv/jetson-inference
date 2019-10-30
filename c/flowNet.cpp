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
 
#include "flowNet.h"
#include "imageNet.cuh"

#include "commandLine.h"


// constructor
flowNet::flowNet() : tensorNet()
{
	mNetworkType = CUSTOM;
}


// destructor
flowNet::~flowNet()
{

}


// NetworkTypeFromStr
flowNet::NetworkType flowNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return flowNet::CUSTOM;

	flowNet::NetworkType type = flowNet::FLOW_320x240;

	// ONNX models
	if( strcasecmp(modelName, "flow-320x240") == 0 || strcasecmp(modelName, "flow-320x240") == 0 || strcasecmp(modelName, "flow_320x240") == 0 || strcasecmp(modelName, "optical-flow-320x240") == 0 || strcasecmp(modelName, "320x240") == 0 || strcasecmp(modelName, "flow") == 0 )
		type = flowNet::FLOW_320x240;
	else if( strcasecmp(modelName, "flow-512x384") == 0 || strcasecmp(modelName, "flow_512x384") == 0 || strcasecmp(modelName, "optical-flow-512x384") == 0 || strcasecmp(modelName, "512x384") == 0 )
		type = flowNet::FLOW_512x384;
	else if( strcasecmp(modelName, "flow-640x480") == 0 || strcasecmp(modelName, "flow_640x480") == 0 || strcasecmp(modelName, "optical-flow-640x480") == 0 || strcasecmp(modelName, "640x480") == 0 )
		type = flowNet::FLOW_640x480;
	else
		type = flowNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* flowNet::NetworkTypeToStr( flowNet::NetworkType type )
{
	switch(type)
	{
		case FLOW_320x240:	return "Optical-Flow-320x240";
		case FLOW_512x384:	return "Optical-Flow-512x384";
		case FLOW_640x480:	return "Optical-Flow-640x480";
		default:			return "Custom";
	}
}


// Create
flowNet* flowNet::Create( flowNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	flowNet* net = NULL;
	
	if( networkType == FLOW_320x240 )
		net = Create("networks/FlowNet-320x240/flownets.onnx", FLOWNET_DEFAULT_INPUT, FLOWNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == FLOW_512x384 )
		net = Create("networks/FlowNet-512x384/flownets.onnx", FLOWNET_DEFAULT_INPUT, FLOWNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);	
	else if( networkType == FLOW_640x480 )
		net = Create("networks/FlowNet-640x480/flownets.onnx", FLOWNET_DEFAULT_INPUT, FLOWNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);

	if( net != NULL )
		net->mNetworkType = networkType;

	return net;
}


// Create
flowNet* flowNet::Create( const char* model_path, const char* input_blob, const char* output_blob,
					 uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	flowNet* net = new flowNet();
	
	if( !net )
		return NULL;
	
	printf("\n");
	printf("flowNet -- loading optical flow network model from:\n");
	printf("         -- model:      %s\n", model_path);
	printf("         -- input_blob  '%s'\n", input_blob);
	printf("         -- output_blob '%s'\n", output_blob);
	printf("         -- batch_size  %u\n\n", maxBatchSize);
	
	// load network
	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("flowNet -- failed to initialize.\n");
		return NULL;
	}

	// load colormaps
	CUDA(cudaColormapInit());

	// return network
	return net;
}


// Create
flowNet* flowNet::Create( int argc, char** argv )
{
	flowNet* net = NULL;

	// obtain the network name
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "flow-320x240");
	
	// enable verbose mode if desired
	if( cmdLine.GetFlag("verbose") )
		tensorNet::EnableVerbose();

	// parse the network type
	const flowNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == flowNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input  = FLOWNET_DEFAULT_INPUT;
		if( !output )  output = FLOWNET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = flowNet::Create(modelName, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = flowNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// Process
bool flowNet::Process( float* prev_image, float* next_image, uint32_t width, uint32_t height )
{
	if( !prev_image || !next_image || width == 0 || height == 0 )
	{
		printf(LOG_TRT "flowNet::Process( 0x%p, 0x%p, %u, %u ) -> invalid parameters\n", prev_image, next_image, width, height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	const uint32_t inputWidth = GetInputWidth();
	const uint32_t inputHeight = GetInputHeight();

	if( IsModelType(MODEL_ONNX) )
	{
		const float2 range  = make_float2(0.0f, 1.0f);			// remap from [0,255] -> [0,1]
		const float3 mean   = make_float3(0.411f, 0.432f, 0.45f);	// mean pixel subtraction
		const float3 stdDev = make_float3(1.0f, 1.0f, 1.0f);		// no std dev applied

		// the previous frame gets put in channels 0,1,2
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)prev_image, width, height, 
										   mInputs[0].CUDA, inputWidth, inputHeight, 
										   range, mean, stdDev, GetStream())) )
		{
			printf(LOG_TRT "flowNet::Process() -- failed to pre-process prev_image\n");
			return false;
		}

		// the next frame gets put in channels 3,4,5
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)next_image, width, height, 
										   mInputs[0].CUDA + inputWidth * inputHeight * 3, 
										   inputWidth, inputHeight,
										   range, mean, stdDev, GetStream())) )
		{
			printf(LOG_TRT "flowNet::Process() -- failed to pre-process next_image\n");
			return false;
		}
	}
	else
	{
		printf(LOG_TRT "flowNet::Process() -- support for models other than ONNX not implemented.\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
	// process with TensorRT
	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	return true;
}


// Process
bool flowNet::Process( float* prev_image, float* next_image, float* flow_map, 
		             uint32_t width, uint32_t height, cudaFilterMode filter )
{
	return Process(prev_image, next_image, width, height, flow_map, width, height, filter);
} 


// Process
bool flowNet::Process( float* prev_image, float* next_image, uint32_t width, uint32_t height,
			        float* flow_map, uint32_t flow_map_width, uint32_t flow_map_height,
			        cudaFilterMode flow_map_filter )
{
	if( !Process(prev_image, next_image, width, height) )
		return false;

	if( !Visualize(flow_map, flow_map_width, flow_map_height, flow_map_filter) )
		return false;

	return true;
}


// Visualize
bool flowNet::Visualize( float* flow_map, uint32_t width, uint32_t height, cudaFilterMode filter )
{
	if( !flow_map || width == 0 || height == 0 )
	{
		printf(LOG_TRT "flowNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", flow_map, width, height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// get the 
	float* flow_field = mOutputs[0].CPU;
	float2 flow_range = make_float2(100000000.0f, -100000000.0f);

	const uint32_t flow_width = GetFlowFieldWidth();
	const uint32_t flow_height = GetFlowFieldHeight();

	for( uint32_t c=0; c < 2; c++ )
	{
		for( uint32_t y=0; y < flow_height; y++ )
		{
			for( int x=0; x < flow_width; x++ )
			{
				const float flow = flow_field[x];

				if( flow < flow_range.x )
					flow_range.x = flow;

				if( flow > flow_range.y )
					flow_range.y = flow;
			}

			flow_field += flow_width;
		}
	}

	printf("flow range:  %f -> %f\n", flow_range.x, flow_range.y);

	const float min_range = 5.0f;

	if( fabs(flow_range.x) < min_range )
		flow_range.x = -min_range;

	if( fabs(flow_range.y) < min_range )
		flow_range.y = min_range;

	printf("adap range:  %f -> %f\n", flow_range.x, flow_range.y);

	PROFILER_END(PROFILER_POSTPROCESS);
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// apply color mapping to depth image
	if( CUDA_FAILED(cudaColormap(GetFlowField(), flow_width, flow_height,
						    flow_map, width, height, flow_range, COLORMAP_FLOW, 
						    filter, FORMAT_CHW, GetStream())) )
	{
		printf(LOG_TRT "flowNet::Process() -- failed to map flow field with cudaColormap()\n");
		return false; 
	}

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}



