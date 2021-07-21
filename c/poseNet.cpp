/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#include "poseNet.h"
#include "tensorConvert.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"
#include "json.hpp"

#include <fstream>

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"


// constructor
poseNet::poseNet() : tensorNet()
{
	mThreshold = POSENET_DEFAULT_THRESHOLD;
	
	memset(mTopology.links, 0, sizeof(mTopology.links));
	mTopology.numLinks = 0;
}


// destructor
poseNet::~poseNet()
{
	
}


// loadTopology
bool poseNet::loadTopology( const char* json_path, Topology* topology )
{
	nlohmann::json topology_json;
	
	try
	{
		std::ifstream topology_file(json_path);
		topology_file >> topology_json;
	}
	catch (...)
	{
		LogError(LOG_TRT "poseNet -- failed to load topology json from '%s'", json_path);
		return false;
	}
		
	// https://nlohmann.github.io/json/features/arbitrary_types/
	topology->keypoints = topology_json["keypoints"].get<std::vector<std::string>>();
	
	for( size_t n=0; n < topology->keypoints.size(); n++ )
		LogInfo("poseNet topology -- keypoint %zu  %s\n", n, topology->keypoints[n].c_str());
	
	// load skeleton links
	const auto skeleton = topology_json["skeleton"].get<std::vector<std::vector<int>>>();
	
	if( skeleton.size() >= Topology::MAX_LINKS )
	{
		LogError("topology from '%s' has more than the maximum number of skeleton links (%zu, max of %i)\n", json_path, skeleton.size(), Topology::MAX_LINKS);
		return false;
	}
	
	for( size_t n=0; n < skeleton.size(); n++ )
	{
		if( skeleton[n].size() != 2 )
		{
			LogError("invalid skeleton link from topology '%s' (link %zu had %zu entries, expected 2)", json_path, n, skeleton[n].size());
			return false;
		}
		
		printf("poseNet topology -- skeleton link %zu  %i %i\n", n, skeleton[n][0], skeleton[n][1]);
		
		topology->links[n*4+0] = n * 2;
		topology->links[n*4+1] = n * 2 + 1;
		topology->links[n*4+2] = skeleton[n][0] - 1;
		topology->links[n*4+3] = skeleton[n][1] - 1;
		
		topology->numLinks++;
	}
	
	return true;
}


// init
bool poseNet::init( const char* model, const char* topology_path, float threshold, 
				const char* input_blob, const char* cmap_blob, const char* paf_blob, uint32_t maxBatchSize, 
				precisionType precision, deviceType device, bool allowGPUFallback )
{
	LogInfo("\n");
	LogInfo("poseNet -- loading pose estimation model from:\n");
	LogInfo("          -- model        %s\n", CHECK_NULL_STR(model));
	LogInfo("          -- topology     %s\n", CHECK_NULL_STR(topology_path));
	LogInfo("          -- input_blob   '%s'\n", CHECK_NULL_STR(input_blob));
	LogInfo("          -- output_cmap  '%s'\n", CHECK_NULL_STR(cmap_blob));
	LogInfo("          -- output_paf   '%s'\n", CHECK_NULL_STR(paf_blob));
	LogInfo("          -- threshold    %f\n", threshold);
	LogInfo("          -- batch_size   %u\n\n", maxBatchSize);
	
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( cmap_blob != NULL )
		output_blobs.push_back(cmap_blob);

	if( paf_blob != NULL )
		output_blobs.push_back(paf_blob);
	
	// load the topology
	if( !loadTopology(topology_path, &mTopology) )
	{
		LogError(LOG_TRT "postNet -- failed to load topology json from '%s'", topology_path);
		return false;
	}
	
	// assert this is an ONNX model
	if( modelTypeFromPath(model) != MODEL_ONNX )
	{
		LogError(LOG_TRT "poseNet -- only ONNX models are supported.\n");
		return false;
	}
	
	// increase default workspace size
	size_t gpuMemFree = 0;
	size_t gpuMemTotal = 0;
	
	CUDA(cudaMemGetInfo(&gpuMemFree, &gpuMemTotal));

	if( gpuMemTotal <= (2048 << 20) )
		mWorkspaceSize = 512 << 20;
	else
		mWorkspaceSize = 2048 << 20;

	// load the model
	if( !LoadNetwork(NULL, model, NULL, input_blob, output_blobs, 
				  maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "poseNet -- failed to initialize.\n");
		return false;
	}

	// set the specified threshold
	SetThreshold(threshold);

	return true;
}


// Create
poseNet* poseNet::Create( const char* model, const char* topology, float threshold, 
					 const char* input_blob, const char* cmap_blob, const char* paf_blob, 
					 uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	poseNet* net = new poseNet();
	
	if( !net )
		return NULL;

	if( !net->init(model, topology, threshold, input_blob, cmap_blob, paf_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
	{
		return NULL;
	}

	return net;
}


// Create
poseNet* poseNet::Create( NetworkType networkType, float threshold, uint32_t maxBatchSize, 
					 precisionType precision, deviceType device, bool allowGPUFallback )
{
	if( networkType == RESNET18_BODY )
		return Create("networks/Pose-ResNet18-Body/pose-resnet18-body.onnx", "networks/Pose-ResNet18-Body/human_pose.json", threshold, POSENET_DEFAULT_INPUT, POSENET_DEFAULT_CMAP, POSENET_DEFAULT_PAF, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == RESNET18_HAND )
		return Create("networks/Pose-ResNet18-Hand/pose-resnet18-hand.onnx", "networks/Pose-ResNet18-Hand/hand_pose.json", threshold, POSENET_DEFAULT_INPUT, POSENET_DEFAULT_CMAP, POSENET_DEFAULT_PAF, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == DENSENET121_BODY )
		return Create("networks/Pose-DenseNet121-Body/pose-densenet121-body.onnx", "networks/Pose-DenseNet121-Body/human_pose.json", threshold, POSENET_DEFAULT_INPUT, POSENET_DEFAULT_CMAP, POSENET_DEFAULT_PAF, maxBatchSize, precision, device, allowGPUFallback );
	else
		return NULL;
}


// NetworkTypeFromStr
poseNet::NetworkType poseNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return poseNet::CUSTOM;

	poseNet::NetworkType type = poseNet::RESNET18_BODY;

	if( strcasecmp(modelName, "resnet18-body") == 0 || strcasecmp(modelName, "resnet18_body") == 0 )
		type = poseNet::RESNET18_BODY;
	else if( strcasecmp(modelName, "resnet18-hand") == 0 || strcasecmp(modelName, "resnet18_hand") == 0 )
		type = poseNet::RESNET18_HAND;
	else if( strcasecmp(modelName, "densenet121-body") == 0 || strcasecmp(modelName, "densenet121_body") == 0 )
		type = poseNet::DENSENET121_BODY;
	else
		type = poseNet::CUSTOM;

	return type;
}


// Create
poseNet* poseNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
poseNet* poseNet::Create( const commandLine& cmdLine )
{
	poseNet* net = NULL;

	// parse command line parameters
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "resnet18-body");

	float threshold = cmdLine.GetFloat("threshold");
	
	if( threshold == 0.0f )
		threshold = POSENET_DEFAULT_THRESHOLD;
	
	int maxBatchSize = cmdLine.GetInt("batch_size");
	
	if( maxBatchSize < 1 )
		maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

	// parse the model type
	const poseNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == poseNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* out_cmap = cmdLine.GetString("output_cmap");
		const char* out_paf  = cmdLine.GetString("output_paf");
		const char* topology = cmdLine.GetString("topology");

		if( !input ) 	
			input = POSENET_DEFAULT_INPUT;

		if( !out_cmap ) out_cmap = POSENET_DEFAULT_CMAP;
		if( !out_paf )  out_paf  = POSENET_DEFAULT_PAF;

		if( !topology )
		{
			LogError(LOG_TRT "poseNet -- must specifiy --topology json file");
			return NULL;
		}

		net = poseNet::Create(modelName, topology, threshold, input, out_cmap, out_paf, maxBatchSize);
	}
	else
	{
		// create poseNet from pretrained model
		net = poseNet::Create(type, threshold, maxBatchSize);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set overlay alpha value
	//net->SetOverlayAlpha(cmdLine.GetFloat("alpha", POSENET_DEFAULT_ALPHA));

	return net;
}
	

// Process
bool poseNet::Process( void* input, uint32_t width, uint32_t height, imageFormat format, uint32_t overlay )
{
	if( !input || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "poseNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", input, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "poseNet::Process() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                      supported formats are:\n");
		LogError(LOG_TRT "                         * rgb8\n");		
		LogError(LOG_TRT "                         * rgba8\n");		
		LogError(LOG_TRT "                         * rgb32f\n");		
		LogError(LOG_TRT "                         * rgba32f\n");

		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaTensorNormMeanRGB(input, format, width, height,
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.485f, 0.456f, 0.406f),
									   make_float3(0.229f, 0.224f, 0.225f),  
									   GetStream())) )
		{
			LogError(LOG_TRT "poseNet::Process() -- cudaTensorNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		LogError(LOG_TRT "poseNet::Process() -- invalid model type (should be ONNX)\n");
		return false;
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
	// execute the model
	if( !ProcessNetwork() )
		return false;
	
	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// TODO post processing
	
	PROFILER_END(PROFILER_POSTPROCESS);

	// TODO overlay
	
	// wait for GPU to complete work			
	//CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return true;
}
