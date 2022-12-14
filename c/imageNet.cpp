/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#include "imageNet.h"
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "logging.h"

#include <algorithm>


// constructor
imageNet::imageNet() : tensorNet()
{
	mNetworkType = CUSTOM;
	mNumClasses  = 0;
	mThreshold   = 0.01f;

	mSmoothingBuffer = NULL;
	mSmoothingFactor = 0;
}


// destructor
imageNet::~imageNet()
{
	CUDA_FREE_HOST(mSmoothingBuffer);
}


// Create
imageNet* imageNet::Create( imageNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(networkType, maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	net->mNetworkType = networkType;
	return net;
}


// Create
imageNet* imageNet::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
					   const char* class_path, const char* input, const char* output, uint32_t maxBatchSize,
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	// check for built-in model string
	NetworkType type = NetworkTypeFromStr(model_path);
	
	if( type != CUSTOM )
		return Create(type, maxBatchSize, precision, device, allowGPUFallback);
		
	// load custom model
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(prototxt_path, model_path, mean_binary, class_path, input, output, maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// init
bool imageNet::init( imageNet::NetworkType networkType, uint32_t maxBatchSize, 
				 precisionType precision, deviceType device, bool allowGPUFallback )
{
	if( networkType == imageNet::ALEXNET )
		return init( "networks/alexnet.prototxt", "networks/bvlc_alexnet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::GOOGLENET )
		return init( "networks/googlenet.prototxt", "networks/bvlc_googlenet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::GOOGLENET_12 )
		return init( "networks/GoogleNet-ILSVRC12-subset/deploy.prototxt", "networks/GoogleNet-ILSVRC12-subset/snapshot_iter_184080.caffemodel", NULL, "networks/GoogleNet-ILSVRC12-subset/labels.txt", IMAGENET_DEFAULT_INPUT, "softmax", maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::RESNET_18 )
		return init( "networks/ResNet-18/deploy.prototxt", "networks/ResNet-18/ResNet-18.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == imageNet::RESNET_50 )
		return init( "networks/ResNet-50/deploy.prototxt", "networks/ResNet-50/ResNet-50.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == imageNet::RESNET_101 )
		return init( "networks/ResNet-101/deploy.prototxt", "networks/ResNet-101/ResNet-101.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::RESNET_152 )
		return init( "networks/ResNet-152/deploy.prototxt", "networks/ResNet-152/ResNet-152.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::VGG_16 )
		return init( "networks/VGG-16/deploy.prototxt", "networks/VGG-16/VGG-16.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::VGG_19 )
		return init( "networks/VGG-19/deploy.prototxt", "networks/VGG-19/VGG-19.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::INCEPTION_V4 )
		return init( "networks/Inception-v4/deploy.prototxt", "networks/Inception-v4/Inception-v4.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else
		return NULL;
}


// init
bool imageNet::init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, 
				const char* input, const char* output, uint32_t maxBatchSize,
				precisionType precision, deviceType device, bool allowGPUFallback )
{
	if( /*!prototxt_path ||*/ !model_path || !class_path || !input || !output )
		return false;

	LogInfo("\n");
	LogInfo("imageNet -- loading classification network model from:\n");
	LogInfo("         -- prototxt     %s\n", prototxt_path);
	LogInfo("         -- model        %s\n", model_path);
	LogInfo("         -- class_labels %s\n", class_path);
	LogInfo("         -- input_blob   '%s'\n", input);
	LogInfo("         -- output_blob  '%s'\n", output);
	LogInfo("         -- batch_size   %u\n\n", maxBatchSize);

	/*
	 * load and parse googlenet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( prototxt_path, model_path, mean_binary, input, output, 
						    maxBatchSize, precision, device, allowGPUFallback ) )
	{
		LogError(LOG_TRT "failed to load %s\n", model_path);
		return false;
	}

	//LogSuccess(LOG_TRT "imageNet -- loaded %s\n", model_path);

	/*
	 * load synset classnames
	 */
	mNumClasses = DIMS_C(mOutputs[0].dims);
	
	if( !loadClassInfo(class_path, mNumClasses) || mClassSynset.size() != mNumClasses || mClassDesc.size() != mNumClasses )
	{
		LogError(LOG_TRT "imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mNumClasses);
		return false;
	}
	
	LogSuccess(LOG_TRT "imageNet -- %s initialized.\n", model_path);
	return true;
}
			

// NetworkTypeFromStr
imageNet::NetworkType imageNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return imageNet::CUSTOM;

	imageNet::NetworkType type = imageNet::GOOGLENET;

	if( strcasecmp(modelName, "alexnet") == 0 )
		type = imageNet::ALEXNET;
	else if( strcasecmp(modelName, "googlenet") == 0 )
		type = imageNet::GOOGLENET;
	else if( strcasecmp(modelName, "googlenet-12") == 0 || strcasecmp(modelName, "googlenet_12") == 0 )
		type = imageNet::GOOGLENET_12;
	else if( strcasecmp(modelName, "resnet-18") == 0 || strcasecmp(modelName, "resnet_18") == 0 || strcasecmp(modelName, "resnet18") == 0 )
		type = imageNet::RESNET_18;
	else if( strcasecmp(modelName, "resnet-50") == 0 || strcasecmp(modelName, "resnet_50") == 0 || strcasecmp(modelName, "resnet50") == 0 )
		type = imageNet::RESNET_50;
	else if( strcasecmp(modelName, "resnet-101") == 0 || strcasecmp(modelName, "resnet_101") == 0 || strcasecmp(modelName, "resnet101") == 0 )
		type = imageNet::RESNET_101;
	else if( strcasecmp(modelName, "resnet-152") == 0 || strcasecmp(modelName, "resnet_152") == 0 || strcasecmp(modelName, "resnet152") == 0 )
		type = imageNet::RESNET_152;
	else if( strcasecmp(modelName, "vgg-16") == 0 || strcasecmp(modelName, "vgg_16") == 0 || strcasecmp(modelName, "vgg16") == 0 )
		type = imageNet::VGG_16;
	else if( strcasecmp(modelName, "vgg-19") == 0 || strcasecmp(modelName, "vgg_19") == 0 || strcasecmp(modelName, "vgg19") == 0 )
		type = imageNet::VGG_19;
	else if( strcasecmp(modelName, "inception-v4") == 0 || strcasecmp(modelName, "inception_v4") == 0 || strcasecmp(modelName, "inceptionv4") == 0 )
		type = imageNet::INCEPTION_V4;
	else
		type = imageNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* imageNet::NetworkTypeToStr( imageNet::NetworkType network )
{
	switch(network)
	{
		case imageNet::ALEXNET:		return "AlexNet";
		case imageNet::GOOGLENET:	return "GoogleNet";
		case imageNet::GOOGLENET_12:	return "GoogleNet-12";
		case imageNet::RESNET_18:	return "ResNet-18";
		case imageNet::RESNET_50:	return "ResNet-50";
		case imageNet::RESNET_101:	return "ResNet-101";
		case imageNet::RESNET_152:	return "ResNet-152";
		case imageNet::VGG_16:		return "VGG-16";
		case imageNet::VGG_19:		return "VGG-19";
		case imageNet::INCEPTION_V4:	return "Inception-v4";
	}

	return "Custom";
}


// Create
imageNet* imageNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
imageNet* imageNet::Create( const commandLine& cmdLine )
{
	imageNet* net = NULL;

	// obtain the network name
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "googlenet");
	
	// parse the network type
	const imageNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == imageNet::CUSTOM )
	{
		const char* prototxt = cmdLine.GetString("prototxt");
		const char* labels   = cmdLine.GetString("labels");
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input    = IMAGENET_DEFAULT_INPUT;
		if( !output )  output   = IMAGENET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = imageNet::Create(prototxt, modelName, NULL, labels, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = imageNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// parse additional arguments
	net->SetThreshold(cmdLine.GetFloat("threshold", net->GetThreshold()));
	net->SetSmoothing(cmdLine.GetFloat("smoothing", net->GetSmoothing()));
	
	return net;
}

	 
// loadClassInfo
bool imageNet::loadClassInfo( const char* filename, int expectedClasses )
{
	if( !LoadClassLabels(filename, mClassDesc, mClassSynset, expectedClasses) )
		return false;

	mClassPath = filename;	
	return true;
}


// preProcess
bool imageNet::preProcess( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	// verify parameters
	if( !image || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "imageNet::preProcess(0x%p, %u, %u) -> invalid parameters\n", image, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "imageNet::Classify() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                        supported formats are:\n");
		LogError(LOG_TRT "                           * rgb8\n");		
		LogError(LOG_TRT "                           * rgba8\n");		
		LogError(LOG_TRT "                           * rgb32f\n");		
		LogError(LOG_TRT "                           * rgba32f\n");

		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( mNetworkType == imageNet::INCEPTION_V4 )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization
		if( CUDA_FAILED(cudaTensorNormRGB(image, format, width, height,
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
								    make_float2(-1.0f, 1.0f), 
								    GetStream())) )
		{
			LogError(LOG_TRT "imageNet::PreProcess() -- cudaTensorNormRGB() failed\n");
			return false;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaTensorNormMeanRGB(image, format, width, height, 
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.485f, 0.456f, 0.406f),
									   make_float3(0.229f, 0.224f, 0.225f), 
									   GetStream())) )
		{
			LogError(LOG_TRT "imageNet::PreProcess() -- cudaTensorNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		// downsample, convert to band-sequential BGR, and apply mean pixel subtraction 
		if( CUDA_FAILED(cudaTensorMeanBGR(image, format, width, height, 
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f),
								    GetStream())) )
		{
			LogError(LOG_TRT "imageNet::PreProcess() -- cudaTensorMeanBGR() failed\n");
			return false;
		}
	}

	PROFILER_END(PROFILER_PREPROCESS);
	return true;
}

	
// Classify
int imageNet::Classify( void* image, uint32_t width, uint32_t height, imageFormat format, float* confidence )
{
	// downsample and convert to band-sequential BGR
	if( !preProcess(image, width, height, format) )
	{
		LogError(LOG_TRT "imageNet::Classify() -- image pre-processing failed\n");
		return -2;
	}
	
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return -2;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// determine the maximum class
	int classIndex = -1;
	float classMax = 0.0f;
	
	float* outputs = applySmoothing();

	for( size_t n=0; n < mNumClasses; n++ )
	{
		const float conf = outputs[n];
		
		if( conf < mThreshold )
			continue;
		
		//LogDebug("class %04zu - %f  (%s)\n", n, conf, mClassDesc[n].c_str());
	
		if( conf > classMax )
		{
			classIndex = n;
			classMax   = conf;
		}
	}
	
	if( confidence != NULL )
		*confidence = classMax;
	
	PROFILER_END(PROFILER_POSTPROCESS);	
	return classIndex;
}

			
// Classify
int imageNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence, imageFormat format )
{
	return Classify(rgba, width, height, format, confidence);
}


// Classify
int imageNet::Classify( void* image, uint32_t width, uint32_t height, imageFormat format, imageNet::Classifications& pred, int topK )
{	
	// downsample and convert to band-sequential BGR
	if( !preProcess(image, width, height, format) )
	{
		LogError(LOG_TRT "imageNet::Classify() -- image pre-processing failed\n");
		return -2;
	}
	
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return -2;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	float* outputs = applySmoothing();
	
	for( uint32_t n=0; n < mNumClasses; n++ )
	{
		const float conf = outputs[n];
		
		if( conf >= mThreshold )
			pred.push_back(std::pair<uint32_t, float>(n, conf));
	}

	std::sort(pred.begin(), pred.end(), [](std::pair<uint32_t, float> &left, std::pair<uint32_t, float> &right) { return left.second > right.second; });

	if( topK > 0 && pred.size() > topK )
		pred.erase(pred.begin() + topK, pred.end());
	
	PROFILER_END(PROFILER_POSTPROCESS);	
	
	if( pred.size() == 0 )
		return -1;

	return pred[0].first;
}


// applySmoothing
float* imageNet::applySmoothing()
{
	float factor = (mSmoothingFactor > 1) ? (1.0f / mSmoothingFactor) : mSmoothingFactor;
	
	if( factor <= 0 || factor >= 1 )
		return mOutputs[0].CPU;
	
	if( !mSmoothingBuffer )
	{
		// allocate the buffer used to accumulate the average outputs
		if( !cudaAllocMapped(&mSmoothingBuffer, mOutputs[0].size) )
		{
			LogWarning(LOG_TRT "imageNet -- failed to allocate smoothing buffer, reverting to raw network outputs");
			return mOutputs[0].CPU;
		}
		
		// initialize from the existing outputs on the first iteration
		memcpy(mSmoothingBuffer, mOutputs[0].CPU, mOutputs[0].size);
		return mSmoothingBuffer;
	}
	
	// https://ucexperiment.wordpress.com/2013/06/10/moving-rolling-and-running-average/
	for( uint32_t n=0; n < mNumClasses; n++ )
		mSmoothingBuffer[n] += factor * (mOutputs[0].CPU[n] - mSmoothingBuffer[n]);  

	//for( uint32_t n=0; n < mNumClasses; n++ )
	//	mSmoothingBuffer[n] = mOutputs[0].CPU[n] * factor + mSmoothingBuffer[n] * (1.0f - factor);
	
	//for( size_t n=0; n < mNumClasses; n++ )
	//	mSmoothingBuffer[n] = ((mSmoothingBuffer[n] * ((1.0f / factor) - 1)) + mOutputs[0].CPU[n]) * factor;

	return mSmoothingBuffer;
}