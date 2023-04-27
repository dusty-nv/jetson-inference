/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "modelDownloader.h"
#include "cudaDraw.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"
#include "json.hpp"

#include "connect_parts.hpp"
#include "find_peaks.hpp"
#include "munkres.hpp"
#include "paf_score_graph.hpp"
#include "refine_peaks.hpp"

#include <fstream>

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"

#define MIN(a,b)  (a < b ? a : b)
#define MAX(a,b)  (a > b ? a : b)
		
#define CMAP 0	 // cmap is output layer 0
#define PAF  1	 // paf is output layer 1


// constructor
poseNet::poseNet() : tensorNet()
{
	mThreshold    = POSENET_DEFAULT_THRESHOLD;
	mPeaks 	    = NULL;
	mPeakCounts   = NULL;
	mRefinedPeaks = NULL;
	mScoreGraph   = NULL;
	mConnections  = NULL;
	mObjects      = NULL;
	mNumObjects   = 0;
	
	mLinkScale 	 = POSENET_DEFAULT_LINK_SCALE;
	mKeypointScale  = POSENET_DEFAULT_KEYPOINT_SCALE;
	mKeypointColors = NULL;
	
	mAssignmentWorkspace = NULL;
	mConnectionWorkspace = NULL;
	
	memset(mTopology.links, 0, sizeof(mTopology.links));
	mTopology.numLinks = 0;
}


// destructor
poseNet::~poseNet()
{
	SAFE_FREE(mPeaks);
	SAFE_FREE(mPeakCounts);
	SAFE_FREE(mRefinedPeaks);
	SAFE_FREE(mScoreGraph);
	SAFE_FREE(mConnections);
	SAFE_FREE(mObjects);
	SAFE_FREE(mAssignmentWorkspace);
	SAFE_FREE(mConnectionWorkspace);
	SAFE_FREE(mKeypointColors);
}


// init
bool poseNet::init( const char* model, const char* topology_path, const char* colors, float threshold, 
				const char* input_blob, const char* cmap_blob, const char* paf_blob, uint32_t maxBatchSize, 
				precisionType precision, deviceType device, bool allowGPUFallback )
{
	LogInfo("\n");
	LogInfo("poseNet -- loading pose estimation model from:\n");
	LogInfo("        -- model        %s\n", CHECK_NULL_STR(model));
	LogInfo("        -- topology     %s\n", CHECK_NULL_STR(topology_path));
	LogInfo("        -- colors       %s\n", CHECK_NULL_STR(colors));
	LogInfo("        -- input_blob   '%s'\n", CHECK_NULL_STR(input_blob));
	LogInfo("        -- output_cmap  '%s'\n", CHECK_NULL_STR(cmap_blob));
	LogInfo("        -- output_paf   '%s'\n", CHECK_NULL_STR(paf_blob));
	LogInfo("        -- threshold    %f\n", threshold);
	LogInfo("        -- batch_size   %u\n\n", maxBatchSize);
	
	
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
	
	// load the keypoint colors
	if( !loadKeypointColors(colors) )
	{
		LogError(LOG_TRT "poseNet -- failed to load keypoint colors, using defaults...\n");
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

	// verify output dimensions
	if( DIMS_W(mOutputs[CMAP].dims) != DIMS_W(mOutputs[PAF].dims) || DIMS_H(mOutputs[CMAP].dims) != DIMS_H(mOutputs[PAF].dims) )
	{
		LogError("poseNet -- cmap dims (C=%i, H=%i, W=%i) don't match paf dims (C=%i, H=%i, W=%i)\n", DIMS_C(mOutputs[CMAP].dims), DIMS_H(mOutputs[CMAP].dims), DIMS_W(mOutputs[CMAP].dims), DIMS_C(mOutputs[PAF].dims), DIMS_H(mOutputs[PAF].dims), DIMS_W(mOutputs[PAF].dims));
		return false;
	}
	
	const int C = DIMS_C(mOutputs[0].dims);
	const int H = DIMS_H(mOutputs[0].dims);
	const int W = DIMS_W(mOutputs[0].dims);
	const int K = mTopology.numLinks;
	const int M = MAX_LINKS;
	
	// alloc post-processing buffers
	mPeaks = (int*)malloc(C * M * 2 * sizeof(int));
	mPeakCounts = (int*)malloc(C * sizeof(int));
	mRefinedPeaks = (float*)malloc(C * M * 2 * sizeof(float));
	
	mScoreGraph = (float*)malloc(K * M * M * sizeof(float));
	mConnections = (int*)malloc(K * M * 2 * sizeof(int));
	mObjects = (int*)malloc(MAX_OBJECTS * C * sizeof(int));
	
	mAssignmentWorkspace = malloc(trt_pose::parse::assignment_out_workspace(M));
	mConnectionWorkspace = malloc(trt_pose::parse::connect_parts_out_workspace(C, M));
	
	if( !mPeaks || !mPeakCounts || !mRefinedPeaks || !mScoreGraph || !mConnections || !mObjects || !mAssignmentWorkspace || !mConnectionWorkspace )
	{
		LogError("poseNet -- failed to allocate post-processing buffers\n");
		return false;
	}

	return true;
}


// Create
poseNet* poseNet::Create( const char* model, const char* topology, const char* colors, float threshold, 
					 const char* input_blob, const char* cmap_blob, const char* paf_blob, 
					 uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	// check for built-in model string
	if( FindModel(POSENET_MODEL_TYPE, model) )
	{
		return Create(model, threshold, maxBatchSize, precision, device, allowGPUFallback);
	}
	else if( fileExtension(model).length() == 0 )
	{
		LogError(LOG_TRT "couldn't find built-in pose estimation model '%s'\n", model);
		return NULL;
	}
	
	// load custom model
	poseNet* net = new poseNet();
	
	if( !net )
		return NULL;

	if( !net->init(model, topology, colors, threshold, input_blob, cmap_blob, paf_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
	{
		return NULL;
	}

	return net;
}


// Create
poseNet* poseNet::Create( const char* network, float threshold, uint32_t maxBatchSize, 
					 precisionType precision, deviceType device, bool allowGPUFallback )
{
	nlohmann::json model;
	
	if( !DownloadModel(POSENET_MODEL_TYPE, network, model) )
		return NULL;
	
	std::string model_dir = "networks/" + model["dir"].get<std::string>() + "/";
	std::string model_path = model_dir + JSON_STR(model["model"]);
	std::string topology = model_dir + JSON_STR(model["topology"]);
	std::string colors = model_dir + JSON_STR(model["colors"]);

	std::string input = JSON_STR_DEFAULT(model["input"], POSENET_DEFAULT_INPUT);
	std::string output_cmap = POSENET_DEFAULT_CMAP;
	std::string output_paf = POSENET_DEFAULT_PAF;

	nlohmann::json output = model["output"];
	
	if( output.is_object() )
	{
		if( output["cmap"].is_string() )
			output_cmap = output["cmap"].get<std::string>();
		
		if( output["paf"].is_string() )
			output_cmap = output["paf"].get<std::string>();
	}
	
	return Create(model_path.c_str(), topology.c_str(), colors.c_str(), threshold,
			    input.c_str(), output_cmap.c_str(), output_paf.c_str(), 
			    maxBatchSize, precision, device, allowGPUFallback);
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
	if( !FindModel(POSENET_MODEL_TYPE, modelName) )
	{
		const char* colors   = cmdLine.GetString("colors");
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
			LogError(LOG_TRT "poseNet -- must specifiy --topology json file\n");
			return NULL;
		}

		net = poseNet::Create(modelName, topology, colors, threshold, input, out_cmap, out_paf, maxBatchSize);
	}
	else
	{
		// create poseNet from pretrained model
		net = poseNet::Create(modelName, threshold, maxBatchSize);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set overlay scales
	net->SetKeypointScale(cmdLine.GetFloat("keypoint-scale", POSENET_DEFAULT_KEYPOINT_SCALE));
	net->SetLinkScale(cmdLine.GetFloat("link-scale", POSENET_DEFAULT_LINK_SCALE));

	return net;
}
	

// loadTopology
bool poseNet::loadTopology( const char* json_path, Topology* topology )
{
	const std::string path = locateFile(json_path);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "poseNet -- failed to find topology file %s\n", json_path);
		return false;
	}
	
	// load the json
	nlohmann::json topology_json;
	
	try
	{
		std::ifstream topology_file(path.c_str());
		topology_file >> topology_json;
	}
	catch (...)
	{
		LogError(LOG_TRT "poseNet -- failed to load topology json from '%s'\n", json_path);
		return false;
	}
	
	// https://nlohmann.github.io/json/features/arbitrary_types/
	topology->category = topology_json["supercategory"].get<std::string>();
	topology->keypoints = topology_json["keypoints"].get<std::vector<std::string>>();
	
	for( size_t n=0; n < topology->keypoints.size(); n++ )
		LogInfo(LOG_TRT "topology -- keypoint %zu  %s\n", n, topology->keypoints[n].c_str());
	
	// load skeleton links
	const auto skeleton = topology_json["skeleton"].get<std::vector<std::vector<int>>>();
	
	if( skeleton.size() >= MAX_LINKS )
	{
		LogError(LOG_TRT "topology from '%s' has more than the maximum number of skeleton links (%zu, max of %i)\n", json_path, skeleton.size(), MAX_LINKS);
		return false;
	}
	
	for( size_t n=0; n < skeleton.size(); n++ )
	{
		if( skeleton[n].size() != 2 )
		{
			LogError(LOG_TRT "invalid skeleton link from topology '%s' (link %zu had %zu entries, expected 2)\n", json_path, n, skeleton[n].size());
			return false;
		}
		
		LogInfo(LOG_TRT "topology -- skeleton link %zu  %i %i\n", n, skeleton[n][0], skeleton[n][1]);
		
		topology->links[n*4+0] = n * 2;
		topology->links[n*4+1] = n * 2 + 1;
		topology->links[n*4+2] = skeleton[n][0] - 1;
		topology->links[n*4+3] = skeleton[n][1] - 1;
		
		topology->numLinks++;
	}
	
	return true;
}


// loadKeypointColors
bool poseNet::loadKeypointColors( const char* filename )
{
	// initialize the colors array
	const uint32_t numKeypoints = mTopology.keypoints.size();
	mKeypointColors = (float4*)malloc(numKeypoints * sizeof(float4));
	
	for( uint32_t n=0; n < numKeypoints; n++ )
		mKeypointColors[n] = make_float4(0,255,0,255);
		
	// check if a colors file is to be loaded
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "poseNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "poseNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class colors
	char str[512];
	int  idx = 0;

	while( fgets(str, 512, f) != NULL && idx < numKeypoints )
	{
		const int len = strlen(str);
		
		if( len > 0 )
		{
			if( str[len-1] == '\n' )
				str[len-1] = 0;

			int r = 255;
			int g = 255;
			int b = 255;
			int a = 255;

			sscanf(str, "%i %i %i %i", &r, &g, &b, &a);
			LogVerbose(LOG_TRT "poseNet -- keypoint %02i '%s'  color %i %i %i %i\n", idx, GetKeypointName(idx), r, g, b, a);
			mKeypointColors[idx] = make_float4(r,g,b,a);
			idx++; 
		}
	}
	
	fclose(f);
	
	LogVerbose(LOG_TRT "poseNet -- loaded %i class colors\n", idx);
	
	if( idx == 0 )
		return false;
	
	return true;
}


// Process
bool poseNet::Process( void* input, uint32_t width, uint32_t height, imageFormat format, std::vector<ObjectPose>& poses, uint32_t overlay )
{
	if( !input || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "poseNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", input, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		imageFormatErrorMsg(LOG_TRT, "poseNet::Process()", format);
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

	if( !postProcess(poses, width, height) )
		return false;
	
	PROFILER_END(PROFILER_POSTPROCESS);
	PROFILER_BEGIN(PROFILER_VISUALIZE);
	
	if( !Overlay(input, input, width, height, format, poses, overlay) )
		return false;
	
	// wait for GPU to complete work			
	//CUDA(cudaDeviceSynchronize());

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// Process
bool poseNet::Process( void* input, uint32_t width, uint32_t height, imageFormat format, uint32_t overlay )
{
	std::vector<ObjectPose> poses;
	return Process(input, width, height, format, poses, overlay);
}


// postProcess
bool poseNet::postProcess(std::vector<ObjectPose>& poses, uint32_t width, uint32_t height)
{
	float* cmap = mOutputs[0].CPU;
	float* paf = mOutputs[1].CPU;

	// https://github.com/NVIDIA-AI-IOT/trt_pose/tree/master/trt_pose/parse#terminology
	const int C = DIMS_C(mOutputs[0].dims);
	const int H = DIMS_H(mOutputs[0].dims);
	const int W = DIMS_W(mOutputs[0].dims);
	const int K = mTopology.numLinks;
	const int M = MAX_LINKS;
	const int N = 1;  // batch size = 1
	
	//LogVerbose("cmap C=%i H=%i W=%i\n", C, H, W);
	//LogVerbose("paf  C=%i H=%i W=%i\n", DIMS_C(mOutputs[1].dims), DIMS_H(mOutputs[1].dims), DIMS_W(mOutputs[1].dims));
	
	// find peaks
	trt_pose::parse::find_peaks_out_nchw(
		mPeakCounts, mPeaks, cmap,
		N, C, H, W, M,
		mThreshold, CMAP_WINDOW_SIZE);
		
	// refine peaks
	trt_pose::parse::refine_peaks_out_nchw(
		mRefinedPeaks, mPeakCounts, mPeaks, cmap,
		N, C, H, W, M, CMAP_WINDOW_SIZE);
	
	// compute score graph
	trt_pose::parse::paf_score_graph_out_nkhw(
		mScoreGraph, mTopology.links, 
		paf, mPeakCounts, mRefinedPeaks,
		N, K, C, H, W, M,
		PAF_INTEGRAL_SAMPLES);
	
	// generate connections
	memset(mConnections, -1, K * M * 2 * sizeof(int));
	memset(mObjects, -1, MAX_OBJECTS * C * sizeof(int));
	
	trt_pose::parse::assignment_out_nk(
		mConnections, mScoreGraph, mTopology.links, mPeakCounts,
		N, C, K, M, mThreshold, mAssignmentWorkspace);
		
	trt_pose::parse::connect_parts_out_batch(
		&mNumObjects, mObjects, mConnections, mTopology.links, mPeakCounts,
		N, K, C, M, MAX_OBJECTS, mConnectionWorkspace);
		
	// collate results
	for( int i=0; i < mNumObjects; i++ )
	{
		ObjectPose obj_pose;
		
		obj_pose.ID     = i;
		obj_pose.Left   = 9999999;
		obj_pose.Top    = 9999999;
		obj_pose.Right  = 0;
		obj_pose.Bottom = 0;
		
		// add valid keypoints
		for( int j=0; j < C; j++ )
		{
			const int k = mObjects[i * C + j];

			if( k >= 0 )
			{
				const int peak_idx = j * M * 2 + k * 2;
				
				ObjectPose::Keypoint keypoint;
				
				keypoint.ID = j;
				keypoint.x  = mRefinedPeaks[peak_idx + 1] * width;
				keypoint.y  = mRefinedPeaks[peak_idx + 0] * height;
				
				obj_pose.Keypoints.push_back(keypoint);
			}
		}
			
		// add valid links
		for( int k=0; k < K; k++ )
		{
			const int c_a = mTopology.links[k * 4 + 2];
			const int c_b = mTopology.links[k * 4 + 3];
			
			const int obj_a = mObjects[i * C + c_a];
			const int obj_b = mObjects[i * C + c_b];
			
			if( obj_a >= 0 && obj_b >= 0 )
			{
				int a = obj_pose.FindKeypoint(c_a);
				int b = obj_pose.FindKeypoint(c_b);
				
				if( a < 0 || b < 0 )
				{
					LogError(LOG_TRT "poseNet::postProcess() -- missing keypoint in output object pose, skipping...\n");
					continue;
				}
				
				const int link_idx = obj_pose.FindLink(a,b);
				
				if( link_idx >= 0 )
				{
					LogWarning(LOG_TRT "poseNet::postProcess() -- duplicate link detected, skipping...\n");
					continue;
				}
				
				if( a > b )
				{
					const int c = a;
					a = b;
					b = c;
				}
				
				obj_pose.Links.push_back({(uint32_t)a, (uint32_t)b});
			}
		}
		
		// get bounding box
		const uint32_t numKeypoints = obj_pose.Keypoints.size();
		
		if( numKeypoints < 2 )
			continue;
		
		for( uint32_t n=0; n < numKeypoints; n++ )
		{
			obj_pose.Left   = MIN(obj_pose.Keypoints[n].x, obj_pose.Left);
			obj_pose.Top    = MIN(obj_pose.Keypoints[n].y, obj_pose.Top);
			obj_pose.Right  = MAX(obj_pose.Keypoints[n].x, obj_pose.Right);
			obj_pose.Bottom = MAX(obj_pose.Keypoints[n].y, obj_pose.Bottom);
		}
		
		poses.push_back(obj_pose);
	}
	
	return true;
}


// Overlay
bool poseNet::Overlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, const std::vector<ObjectPose>& poses, uint32_t overlay )
{
	if( !input || !output || width == 0 || height == 0 )
		return false;
	
	if( overlay == OVERLAY_NONE )
		return true;
	
	if( input != output )
	{
		if( CUDA_FAILED(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice)) )
		{
			LogError(LOG_TRT "poseNet -- Overlay() failed to copy input image to output image\n");
			return false;
		}
	}
	
	const uint32_t numObjects = poses.size();
	
	const float line_width = MAX(MAX(width,height) * mLinkScale, 1.5f);
	const float circle_radius = MAX(MAX(width,height) * mKeypointScale, 4.0f);
	
	for( uint32_t o=0; o < numObjects; o++ )
	{
		if( overlay & OVERLAY_BOX )
		{
			CUDA(cudaDrawRect(input, output, width, height, format,
						   poses[o].Left, poses[o].Top, poses[o].Right, poses[o].Bottom,
						   make_float4(255, 255, 255, 100)));
		}
		
		if( overlay & OVERLAY_LINKS )
		{
			const uint32_t numLinks = poses[o].Links.size();
			
			for( uint32_t l=0; l < numLinks; l++ )
			{
				const uint32_t a = poses[o].Links[l][0];
				const uint32_t b = poses[o].Links[l][1];
				
				CUDA(cudaDrawLine(input, output, width, height, format, 
							   poses[o].Keypoints[a].x, poses[o].Keypoints[a].y, 
							   poses[o].Keypoints[b].x, poses[o].Keypoints[b].y,
							   mKeypointColors[poses[o].Keypoints[b].ID], line_width));
			}
		}
		
		if( overlay & OVERLAY_KEYPOINTS )
		{
			const uint32_t numKeypoints = poses[o].Keypoints.size();
						
			for( uint32_t k=0; k < numKeypoints; k++ )
			{
				CUDA(cudaDrawCircle(input, output, width, height, format, 
								poses[o].Keypoints[k].x, poses[o].Keypoints[k].y, 
								circle_radius, mKeypointColors[poses[o].Keypoints[k].ID]));
			}
		}
	}
	
	return true;
}


// OverlayFlagsFromStr
uint32_t poseNet::OverlayFlagsFromStr( const char* str_user )
{
	if( !str_user )
		return OVERLAY_DEFAULT;

	// copy the input string into a temporary array,
	// because strok modifies the string
	const size_t str_length = strlen(str_user);
	const size_t max_length = 256;
	
	if( str_length == 0 )
		return OVERLAY_DEFAULT;

	if( str_length >= max_length )
	{
		LogError(LOG_TRT "poseNet::OverlayFlagsFromStr() overlay string exceeded max length of %zu characters ('%s')", max_length, str_user);
		return OVERLAY_DEFAULT;
	}
	
	char str[max_length];
	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
		return OVERLAY_DEFAULT;

	// look for the tokens:  "keypoints", "links", "default", and "none"
	uint32_t flags = OVERLAY_NONE;

	while( token != NULL )
	{
		if( strcasecmp(token, "keypoints") == 0 || strcasecmp(token, "keypoint") == 0 )
			flags |= OVERLAY_KEYPOINTS;
		else if( strcasecmp(token, "links") == 0 || strcasecmp(token, "link") == 0 )
			flags |= OVERLAY_LINKS;
		else if( strcasecmp(token, "box") == 0 || strcasecmp(token, "boxes") == 0 )
			flags |= OVERLAY_BOX;
		else if( strcasecmp(token, "default") == 0 )
			flags |= OVERLAY_DEFAULT;
		
		token = strtok(NULL, delimiters);
	}	

	return flags;
}