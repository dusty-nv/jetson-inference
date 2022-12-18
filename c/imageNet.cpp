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
#include "modelDownloader.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"

#include <algorithm>



// constructor
imageNet::imageNet() : tensorNet()
{
	mThreshold = IMAGENET_DEFAULT_THRESHOLD;
	mNumClasses = 0;
	
	mSmoothingBuffer = NULL;
	mSmoothingFactor = 0;
}


// destructor
imageNet::~imageNet()
{
	CUDA_FREE_HOST(mSmoothingBuffer);
}

	
// Create
imageNet* imageNet::Create( const char* network, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	nlohmann::json model;
	
	if( !DownloadModel(IMAGENET_MODEL_TYPE, network, model) )
		return NULL;
	
	std::string model_dir = "networks/" + model["dir"].get<std::string>() + "/";
	std::string model_path = model_dir + model["model"].get<std::string>();
	std::string prototxt = JSON_STR(model["prototxt"]);
	std::string labels = JSON_STR(model["labels"]);
	std::string input = JSON_STR_DEFAULT(model["input"], IMAGENET_DEFAULT_INPUT);
	std::string output = JSON_STR_DEFAULT(model["output"], IMAGENET_DEFAULT_OUTPUT);
		
	if( prototxt.length() > 0 )
		prototxt = model_dir + prototxt;
	
	if( locateFile(labels).length() == 0 )
		labels = model_dir + labels;
	
	return Create(prototxt.c_str(), model_path.c_str(), NULL, 
			    labels.c_str(), input.c_str(), output.c_str(), 
			    maxBatchSize, precision, device, allowGPUFallback);
}

	
// Create
imageNet* imageNet::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
					   const char* class_path, const char* input, const char* output, uint32_t maxBatchSize,
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	// check for built-in model string
	if( FindModel(IMAGENET_MODEL_TYPE, model_path) )
	{
		return Create(model_path, maxBatchSize, precision, device, allowGPUFallback);
	}
	else if( fileExtension(model_path).length() == 0 )
	{
		LogError(LOG_TRT "couldn't find built-in classification model '%s'\n", model_path);
		return NULL;
	}
		
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
	if( !FindModel(IMAGENET_MODEL_TYPE, modelName) )
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
		net = imageNet::Create(modelName);
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

	if( mModelFile == "Inception-v4.caffemodel" )
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