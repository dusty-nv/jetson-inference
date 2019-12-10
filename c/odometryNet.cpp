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
 
#include "odometryNet.h"

#include "commandLine.h"
#include "cudaUtility.h"


// constructor
odometryNet::odometryNet() : tensorNet()
{
	mNetworkType = CUSTOM;
}


// destructor
odometryNet::~odometryNet()
{

}


// NetworkTypeFromStr
odometryNet::NetworkType odometryNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return odometryNet::CUSTOM;

	odometryNet::NetworkType type = odometryNet::CUSTOM;

	if( strcasecmp(modelName, "resnet18-tum") == 0 || strcasecmp(modelName, "resnet18_tum") == 0 || strcasecmp(modelName, "tum") == 0 )
		type = odometryNet::RESNET18_TUM;
	else if( strcasecmp(modelName, "resnet18-cooridor") == 0 || strcasecmp(modelName, "resnet18_cooridor") == 0 || strcasecmp(modelName, "cooridor") == 0 )
		type = odometryNet::RESNET18_COORIDOR;
	else
		type = odometryNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* odometryNet::NetworkTypeToStr( odometryNet::NetworkType type )
{
	switch(type)
	{
		// ONNX models
		case RESNET18_TUM:			return "resnet18-tum";
		case RESNET18_COORIDOR:		return "resnet18-cooridor";
		default:					return "custom odometryNet";
	}
}


// Create
odometryNet* odometryNet::Create( odometryNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	odometryNet* net = NULL;

	if( networkType == RESNET18_TUM )
		net = Create("networks/OdometryNet-ResNet18-TUM/resnet18.onnx", ODOMETRY_NET_DEFAULT_INPUT, ODOMETRY_NET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == RESNET18_COORIDOR )
		net = Create("networks/OdometryNet-ResNet18-Cooridor/resnet18.onnx", ODOMETRY_NET_DEFAULT_INPUT, ODOMETRY_NET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);

	if( !net )
		return NULL;

	net->mNetworkType = networkType;
}


// Create
odometryNet* odometryNet::Create( const char* model_path, const char* input, 
						    const char* output, uint32_t maxBatchSize,
					   	    precisionType precision, deviceType device, 
						    bool allowGPUFallback )
{
	if( !model_path || !input || !output )
		return NULL;

	printf("\n");
	printf("odometryNet -- loading homography network model from:\n");
	printf("            -- model        %s\n", model_path);
	printf("            -- input_blob   '%s'\n", input);
	printf("            -- output_blob  '%s'\n", output);
	printf("            -- batch_size   %u\n\n", maxBatchSize);

	// create the homography network
	odometryNet* net = new odometryNet();
	
	if( !net )
		return NULL;
	
	// load the model
	if( !net->LoadNetwork(NULL, model_path, NULL,
					  input, output, maxBatchSize,
					  precision, device, allowGPUFallback) )
	{
		printf(LOG_TRT "failed to load odometryNet\n");
		delete net;
		return NULL;
	}
	
	printf(LOG_TRT "%s loaded\n", model_path);
	return net;
}


// Create
odometryNet* odometryNet::Create( int argc, char** argv )
{
	odometryNet* net = NULL;	

	// enable verbose mode if desired
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("verbose") )
		tensorNet::EnableVerbose();

	// parse the desired model type
	const char* model = cmdLine.GetString("network");

	if( !model )
		model = cmdLine.GetString("model", "resnet18-tum");

	// load the odometry model
	odometryNet::NetworkType type = NetworkTypeFromStr(model);

	if( type == odometryNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input )  input  = ODOMETRY_NET_DEFAULT_INPUT;
		if( !output ) output = ODOMETRY_NET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = 1;

		net = odometryNet::Create(model, input, output, maxBatchSize/*, TYPE_FP32*/);
	}
	else
	{
		// create from pretrained model
		net = odometryNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// from odometryNet.cu
cudaError_t cudaPreOdometryNet( float4* inputA, float4* inputB, size_t inputWidth, size_t inputHeight,
				         	  float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );

// Process
bool odometryNet::Process( float4* imageA, float4* imageB, uint32_t width, uint32_t height, float* output )
{
	if( !imageA || !imageB || width == 0 || height == 0 )
	{
		printf(LOG_TRT "odometryNet::Process() -- invalid user inputs\n");
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	/*
	 * convert/rescale the individual RGBA images into grayscale planar format
	 */
	if( CUDA_FAILED(cudaPreOdometryNet(imageA, imageB, width, height,
								mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
								GetStream())) )
	{
		printf(LOG_TRT "odometryNet::Process() -- cudaPreOdometryNet() failed\n");
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

	/*
	 * copy the outputs (optional)
	 */
	if( output != NULL )
		memcpy(output, GetOutput(), GetNumOutputs() * sizeof(float));

	return true;
}


