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
#include "stereoNet.h"
#include "imageNet.cuh"

#include "filesystem.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <unordered_map>

#include "../../plugins/stereo/redtail_tensorrt_plugins.h"
#include "../../plugins/stereo/networks.h"



// constructor
stereoNet::stereoNet() : tensorNet()
{
	mNetworkType = DEFAULT_NETWORK;
}


// destructor
stereoNet::~stereoNet()
{

}


// NetworkTypeFromStr
stereoNet::NetworkType stereoNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return stereoNet::DEFAULT_NETWORK;

	stereoNet::NetworkType type = stereoNet::DEFAULT_NETWORK;

	if( strcasecmp(modelName, "nvsmall") == 0 || strcasecmp(modelName, "nv-small") == 0 || strcasecmp(modelName, "nv_small") == 0 )
		type = stereoNet::NV_SMALL;
	else if( strcasecmp(modelName, "nvtiny") == 0 || strcasecmp(modelName, "nv-tiny") == 0 || strcasecmp(modelName, "nv_tiny") == 0 )
		type = stereoNet::NV_TINY;
	else if( strcasecmp(modelName, "resnet18") == 0 || strcasecmp(modelName, "resnet-18") == 0 || strcasecmp(modelName, "resnet_18") == 0 )
		type = stereoNet::RESNET18;
	else if( strcasecmp(modelName, "resnet18-2d") == 0 || strcasecmp(modelName, "resnet-18-2d") == 0 || strcasecmp(modelName, "resnet_18_2d") == 0 )
		type = stereoNet::RESNET18_2D;
	else
		type = stereoNet::DEFAULT_NETWORK;

	return type;
}


// NetworkTypeToStr
const char* stereoNet::NetworkTypeToStr( stereoNet::NetworkType type )
{
	switch(type)
	{
		case NV_SMALL:		return "NV-Small";
		case NV_TINY:		return "NV-Tiny";
		case RESNET18:		return "ResNet18";
		default:			return "ResNet18_2D";
	}
}


// Create
stereoNet* stereoNet::Create( stereoNet::NetworkType networkType, uint32_t maxBatchSize, 
					     precisionType precision, deviceType device, bool allowGPUFallback )
{
	stereoNet* net = new stereoNet();

	if( !net )
		return NULL;

	if( !net->init(networkType, maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf(LOG_TRT "stereoNet -- failed to load network\n");
		delete net;
		return NULL;
	}

	printf(LOG_TRT "stereoNet -- network intialized\n");
	return net;
}


// readWeights
static std::unordered_map<std::string, nvinfer1::Weights> readWeights(const std::string& filename, nvinfer1::DataType data_type)
{
    assert(data_type == nvinfer1::DataType::kFLOAT || data_type == nvinfer1::DataType::kHALF);

    std::unordered_map<std::string, nvinfer1::Weights> weights;
    std::ifstream weights_file(filename, std::ios::binary);
    assert(weights_file.is_open());
    while (weights_file.peek() != std::ifstream::traits_type::eof())
    {
        std::string name;
        uint32_t    count;
        nvinfer1::Weights w {data_type, nullptr, 0};
        std::getline(weights_file, name, '\0');
        weights_file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        w.count = count;
        size_t el_size_bytes = data_type == nvinfer1::DataType::kFLOAT ? 4 : 2;
        auto p = new uint8_t[count * el_size_bytes];
        weights_file.read(reinterpret_cast<char*>(p), count * el_size_bytes);
        w.values = p;
        assert(weights.find(name) == weights.cend());
        weights[name] = w;
    }
    return weights;
}


// init
bool stereoNet::init( stereoNet::NetworkType networkType, uint32_t maxBatchSize, 
				  precisionType precision, deviceType device, bool allowGPUFallback )
{
	const char* weightsFP32[] = {
		"networks/StereoDNN-NvSmall/trt_weights.bin",
		"networks/StereoDNN-NvTiny/trt_weights.bin",
		"networks/StereoDNN-ResNet18/trt_weights.bin",
		"networks/StereoDNN-ResNet18-2D/trt_weights.bin" };

	const char* weightsFP16[] = {
		"networks/StereoDNN-NvSmall/trt_weights_fp16.bin",
		"networks/StereoDNN-NvTiny/trt_weights_fp16.bin",
		"networks/StereoDNN-ResNet18/trt_weights_fp16.bin",
		"networks/StereoDNN-ResNet18-2D/trt_weights_fp16.bin" };

	// resolve precision type
	precision = SelectPrecision(precision, device, false);

	if( precision != TYPE_FP32 && precision != TYPE_FP16 )
	{
		printf(LOG_TRT "stereoNet only supports FP16/FP32 precision (%s requested)\n", precisionTypeToStr(precision));
		return false;
	}

	// locate the proper weights file
	const std::string weightsPath = locateFile( (precision == TYPE_FP16) ? weightsFP16[networkType] : weightsFP32[networkType] );
	const std::string enginePath = weightsPath + ".engine";

	if( weightsPath.size() == 0 )
	{
		printf(LOG_TRT "stereoNet could not find weights file\n");
		return false;
	}

	printf(LOG_TRT "stereoNet -- loading weights from %s\n", weightsPath.c_str());

	// create the plugins container
	auto pluginContainer = redtail::tensorrt::IPluginContainer::create(gLogger);

	std::vector<std::string> input_layers;
	std::vector<std::string> output_layers;

	input_layers.push_back(STEREONET_DEFAULT_INPUT_LEFT);
	input_layers.push_back(STEREONET_DEFAULT_INPUT_RIGHT);

	output_layers.push_back(STEREONET_DEFAULT_OUTPUT);


	// check if we can load pre-built model from TRT plan file.
	// currently only ResNet18_2D supports serialization.
 	if( networkType == RESNET18_2D && fileExists(enginePath.c_str()) )
	{
		redtail::tensorrt::StereoDnnPluginFactory pluginFactory(*pluginContainer);

		if( !LoadEngine(enginePath.c_str(), input_layers, output_layers, &pluginFactory, device) )
			return false;
	}
	else
	{
		// load the weights from disk
		const nvinfer1::DataType dataType = (precision == TYPE_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;		
		auto weights = readWeights(weightsPath, dataType);

		// create builder and network
		nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
		nvinfer1::INetworkDefinition* network = NULL;

		// for now only ResNet18_2D has proper support for FP16
		if( networkType == NV_SMALL )
			network = createNVSmall1025x321Network(*builder, *pluginContainer, nvinfer1::DimsCHW { 3, 321, 1025 }, weights, nvinfer1::DataType::kFLOAT, gLogger);
		else if( networkType == NV_TINY )
			network = createNVTiny513x161Network(*builder, *pluginContainer, nvinfer1::DimsCHW { 3, 161, 513 }, weights, nvinfer1::DataType::kFLOAT, gLogger);
		else if( networkType == RESNET18 )
			network = createResNet18_1025x321Network(*builder, *pluginContainer, nvinfer1::DimsCHW { 3, 321, 1025 }, weights, nvinfer1::DataType::kFLOAT, gLogger);
		else if( networkType == RESNET18_2D )
			network = createResNet18_2D_513x257Network(*builder, *pluginContainer, nvinfer1::DimsCHW { 3, 257, 513 }, weights, dataType, gLogger);

		if( !network )
		{
			printf(LOG_TRT "stereoNet failed to create network definition\n");
			return false;
		}

		// configure the builder
		if( !ConfigureBuilder(builder, maxBatchSize, 1024 * 1024 * 1024,
						  precision, device, allowGPUFallback, NULL) )
		{
			printf(LOG_TRT "device %s, failed to configure builder\n", deviceTypeToStr(device));
			return false;
		}


		// build the CUDA engine
		printf(LOG_TRT "device %s, building CUDA engine (this may take a few minutes)\n", deviceTypeToStr(device));

		nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);

		if( !engine )
		{
			printf(LOG_TRT "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
			return false;
		}

		printf(LOG_TRT "device %s, completed building CUDA engine\n", deviceTypeToStr(device));

		// only ResNet18_2D supports serialization
		if( networkType == RESNET18_2D )
		{
			// serialize the engine
			nvinfer1::IHostMemory* serMem = engine->serialize();

			if( !serMem )
			{
				printf(LOG_TRT "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
				return false;
			}

			const char* serData = (char*)serMem->data();
			const size_t serSize = serMem->size();

			// write the engine file
			FILE* cacheFile = NULL;
			cacheFile = fopen(enginePath.c_str(), "wb");

			if( cacheFile != NULL )
			{
				if( fwrite(serData,	1, serSize, cacheFile) != serSize )
					printf(LOG_TRT "failed to write %zu bytes to engine cache file %s\n", serSize, enginePath.c_str());
			
				fclose(cacheFile);
			}
			else
			{
				printf(LOG_TRT "failed to open engine cache file for writing %s\n", enginePath.c_str());
			}
		}

		// free builder resources
		network->destroy();
		builder->destroy();

		// finish creating engine
		if( !LoadEngine(engine, input_layers, output_layers, device) )
			return false;
	}

	mPrecision = precision;
	mNetworkType = networkType;

	return true;
}


// Create
/*stereoNet* stereoNet::Create( const char* model_path, const char* input_blob, const char* output_blob,
					   uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	stereoNet* net = new stereoNet();
	
	if( !net )
		return NULL;
	
	printf("\n");
	printf("stereoNet -- loading mono depth network model from:\n");
	printf("         -- model:      %s\n", model_path);
	printf("         -- input_blob  '%s'\n", input_blob);
	printf("         -- output_blob '%s'\n", output_blob);
	printf("         -- batch_size  %u\n\n", maxBatchSize);
	
	// load network
	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("stereoNet -- failed to initialize.\n");
		return NULL;
	}

	// load colormaps
	CUDA(cudaColormapInit());

	// return network
	return net;
}*/


// Create
/*stereoNet* stereoNet::Create( int argc, char** argv )
{
	stereoNet* net = NULL;

	// obtain the network name
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "mobilenet");
	
	// parse the network type
	const stereoNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == stereoNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input  = DEPTHNET_DEFAULT_INPUT;
		if( !output )  output = DEPTHNET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = stereoNet::Create(modelName, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = stereoNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}*/



// Process
bool stereoNet::Process( float* left, float* right, uint32_t input_width, uint32_t input_height )
{
	if( !left || !right || input_width == 0 || input_height == 0 )
	{
		printf(LOG_TRT "stereoNet::Process( %p, %p, %u, %u ) -> invalid parameters\n", left, right, input_width, input_height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	// remap from [0,255] -> [0,1], no mean pixel subtraction or std dev applied
	const float2 range  = make_float2(0.0f, 1.0f);
	const float3 mean   = make_float3(0.0f, 0.0f, 0.0f);
	const float3 stdDev = make_float3(1.0f, 1.0f, 1.0f);

	if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)left, input_width, input_height, 
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									   range, mean, stdDev, GetStream())) )
	{
		printf(LOG_TRT "stereoNet::Process() -- cudaPreImageNetNormMeanRGB() failed for left image\n");
		return false;
	}

	if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)right, input_width, input_height, 
									   mInputs[1].CUDA, GetInputWidth(), GetInputHeight(), 
									   range, mean, stdDev, GetStream())) )
	{
		printf(LOG_TRT "stereoNet::Process() -- cudaPreImageNetNormMeanRGB() failed for right image\n");
		return false;
	}


	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
	// process with TensorRT
	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	const int depth_width = GetDepthFieldWidth();
	const int depth_height = GetDepthFieldHeight();

	// find the min/max depth range
	mDepthRange = make_float2(100000000.0f, -100000000.0f);

	for( int y=0; y < depth_height; y++ )
	{
		for( int x=0; x < depth_width; x++ )
		{
			const float depth = mOutputs[0].CPU[y * depth_width + x];

			if( depth < mDepthRange.x )
				mDepthRange.x = depth;

			if( depth > mDepthRange.y )
				mDepthRange.y = depth;
		}
	}

	printf("depth range:  %f -> %f\n", mDepthRange.x, mDepthRange.y);
	//depthRange = make_float2(0.95f, 5.0f);

	PROFILER_END(PROFILER_POSTPROCESS);
	return true;
}


// Process
bool stereoNet::Process( float* left, float* right, float* output, uint32_t width, uint32_t height, cudaColormapType colormap, cudaFilterMode filter )
{
	return Process(left, right, width, height, output, width, height, colormap, filter);
}


// Process
bool stereoNet::Process( float* left, float* right, uint32_t input_width, uint32_t input_height,
				     float* output, uint32_t output_width, uint32_t output_height, 
                         cudaColormapType colormap, cudaFilterMode filter )
{
	if( !Process(left, right, input_width, input_height) )
		return false;

	if( !Visualize(output, output_width, output_height, colormap, filter) )
		return false;

	return true;
}


// Visualize
bool stereoNet::Visualize( float* output, uint32_t output_width, uint32_t output_height,
				 	 cudaColormapType colormap, cudaFilterMode filter )
{
	if( !output || output_width == 0 || output_height == 0 )
	{
		printf(LOG_TRT "stereoNet::Visualize( 0x%p, %u, %u ) -> invalid parameters\n", output, output_width, output_height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// apply color mapping to depth image
	if( CUDA_FAILED(cudaColormap(GetDepthField(), GetDepthFieldWidth(), GetDepthFieldHeight(),
						    output, output_width, output_height, mDepthRange, 
						    colormap, filter, FORMAT_DEFAULT, GetStream())) )
	{
		printf(LOG_TRT "stereoNet::Visualize() -- failed to map depth image with cudaColormap()\n");
		return false; 
	}

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}




