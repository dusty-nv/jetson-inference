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

#include "featureNet.h"
#include "tensorConvert.h"

#include "cudaDraw.h"
#include "cudaFont.h"
#include "cudaResize.h"
#include "cudaNormalize.h"
#include "cudaColorspace.h"
#include "cudaMappedMemory.h"

#include "commandLine.h"
#include "logging.h"


// constructor
featureNet::featureNet() : tensorNet()
{
	mFont        = NULL;
	mInputWidth  = 0;
	mInputHeight = 0;
	mMaxFeatures = 0;
	mResizedImg  = NULL;
	mNetworkType = CUSTOM;
}


// destructor
featureNet::~featureNet()
{
	CUDA_FREE(mResizedImg);
	SAFE_DELETE(mFont);
}


// Create
featureNet* featureNet::Create( featureNet::NetworkType networkType, uint32_t maxBatchSize, 
					     precisionType precision, deviceType device, bool allowGPUFallback )
{
	featureNet* net = NULL;
	
	if( networkType == LOFTR_COARSE )
		net = Create("networks/Feature-LoFTR-Coarse/LoFTR_teacher.onnx", FEATURENET_DEFAULT_INPUT_0, FEATURENET_DEFAULT_INPUT_1, "1026", maxBatchSize, precision, device, allowGPUFallback);
	
	if( !net )
	{
		LogError(LOG_TRT "featureNet -- invalid built-in model '%s' requested\n", featureNet::NetworkTypeToStr(networkType));
		return NULL;
	}
	
	net->mNetworkType = networkType;
	
	return net;
}


// Create
featureNet* featureNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
featureNet* featureNet::Create( const commandLine& cmdLine )
{
	featureNet* net = NULL;

	// obtain the network name
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "loftr-coarse");
	
	// parse the network type
	const featureNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == featureNet::CUSTOM )
	{
		const char* input_0 = cmdLine.GetString("input_blob_0", FEATURENET_DEFAULT_INPUT_0);
		const char* input_1 = cmdLine.GetString("input_blob_1", FEATURENET_DEFAULT_INPUT_1);
		const char* output  = cmdLine.GetString("output_blob", FEATURENET_DEFAULT_OUTPUT);

		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = featureNet::Create(modelName, input_0, input_1, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = featureNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// Create
featureNet* featureNet::Create( const char* model_path, const char* input_0, 
						  const char* input_1, const char* output, 
						  uint32_t maxBatchSize, precisionType precision, 
						  deviceType device, bool allowGPUFallback )
{
	featureNet* net = new featureNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(model_path, input_0, input_1, output, maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;
	
	return net;
}


// init
bool featureNet::init( const char* model_path, const char* input_0, 
				   const char* input_1, const char* output, 
				   uint32_t maxBatchSize, precisionType precision, 
				   deviceType device, bool allowGPUFallback )
{
	if( !model_path || !input_0 || !input_1 || !output )
		return NULL;
	
	LogInfo("\n");
	LogInfo("featureNet -- loading feature matching network model from:\n");
	LogInfo("           -- model        %s\n", model_path);
	LogInfo("           -- input_blob_0 '%s'\n", input_0);
	LogInfo("           -- input_blob_1 '%s'\n", input_1);
	LogInfo("           -- output_blob  '%s'\n", output);
	LogInfo("           -- batch_size   %u\n\n", maxBatchSize);
	
	// create list of input/output layers
	std::vector<std::string> input_blobs;
	std::vector<std::string> output_blobs;
	
	input_blobs.push_back(input_0);
	input_blobs.push_back(input_1);
	
	output_blobs.push_back("1019");   // hack so that all output tensors are allotted
	output_blobs.push_back(output);
	
	// load model
	if( !tensorNet::LoadNetwork(NULL, model_path, NULL, input_blobs, output_blobs, 
					        maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "failed to load %s\n", model_path);
		return false;
	}
	
	mInputWidth  = DIMS_W(mInputs[0].dims);
	mInputHeight = DIMS_H(mInputs[0].dims);
	mMaxFeatures = mOutputs[0].dims.d[1];
	
	// allocate preprocessing memory
	if( CUDA_FAILED(cudaMalloc(&mResizedImg, mInputWidth * mInputHeight * sizeof(float) * 4)) )
		return false;
	
	return true;
}


/*
// init
bool featureNet::init(const char* model_path, const char* class_path, 
				 const char* input, const char* output, 
				 uint32_t maxBatchSize, precisionType precision, 
				 deviceType device, bool allowGPUFallback )
{
	if( !model_path || !class_path || !input || !output )
		return false;

	LogInfo("\n");
	LogInfo("featureNet -- loading action recognition network model from:\n");
	LogInfo("          -- model        %s\n", model_path);
	LogInfo("          -- class_labels %s\n", class_path);
	LogInfo("          -- input_blob   '%s'\n", input);
	LogInfo("          -- output_blob  '%s'\n", output);
	LogInfo("          -- batch_size   %u\n\n", maxBatchSize);

	if( !tensorNet::LoadNetwork(NULL, model_path, NULL, input, output, 
						   maxBatchSize, precision, device, allowGPUFallback ) )
	{
		LogError(LOG_TRT "failed to load %s\n", model_path);
		return false;
	}
	
	// setup input ring buffer
	mInputBuffers[0] = mInputs[0].CUDA;
	
	if( CUDA_FAILED(cudaMalloc((void**)&mInputBuffers[1], mInputs[0].size)) )
		return false;
	
	CUDA(cudaMemset(mInputBuffers[1], 0, mInputs[0].size));

	// load classnames
	mNumFrames = mInputs[0].dims.d[1];
	mNumClasses = mOutputs[0].dims.d[0];

	if( !imageNet::LoadClassInfo(class_path, mClassDesc, mNumClasses) || mClassDesc.size() != mNumClasses )
	{
		LogError(LOG_TRT "featureNet -- failed to load class descriptions  (%zu of %u)\n", mClassDesc.size(), mNumClasses);
		return false;
	}
	
	LogSuccess(LOG_TRT "featureNet -- %s initialized.\n", model_path);
	return true;
}
*/		

// NetworkTypeFromStr
featureNet::NetworkType featureNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return featureNet::CUSTOM;

	featureNet::NetworkType type = featureNet::CUSTOM;

	if( strcasecmp(modelName, "loftr-coarse") == 0 || strcasecmp(modelName, "loftr_coarse") == 0 )
		type = featureNet::LOFTR_COARSE;
	else
		type = featureNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* featureNet::NetworkTypeToStr( featureNet::NetworkType network )
{
	switch(network)
	{
		case featureNet::LOFTR_COARSE:  return "LoFTR-Coarse";
	}

	return "Custom";
}
	
	
// preProcess
bool featureNet::preProcess( void* image, uint32_t width, uint32_t height, imageFormat format, uint32_t binding )
{
	if( CUDA_FAILED(cudaResize(image, width, height, mResizedImg, mInputWidth, mInputHeight, format)) )
		return false;
	
	if( CUDA_FAILED(cudaConvertColor(mResizedImg, format, mInputs[binding].CUDA, IMAGE_GRAY32F, mInputWidth, mInputHeight)) )
		return false;
	
	if( CUDA_FAILED(cudaNormalize(mInputs[binding].CUDA, make_float2(0,255), mInputs[binding].CUDA, make_float2(0,1), mInputWidth, mInputHeight, IMAGE_GRAY32F)) )
		return false;
	
	return true;
}


// Match
int featureNet::Match( void* image_A, uint32_t width_A, uint32_t height_A, imageFormat format_A, 
				   void* image_B, uint32_t width_B, uint32_t height_B, imageFormat format_B, 
				   float2* features_A, float2* features_B, float* confidence, 
				   float threshold, bool sorted )
{
	// verify parameters
	if( !image_A || !image_B || width_A == 0 || height_A == 0 || width_B == 0 || height_B == 0 || !features_A || !features_B || !confidence )
	{
		LogError(LOG_TRT "featureNet::Match() called with NULL / invalid parameters\n");
		return -1;
	}

	// preprocess images
	PROFILER_BEGIN(PROFILER_PREPROCESS);
	
	if( !preProcess(image_A, width_A, height_A, format_A, 0) )
		return -1;
	
	if( !preProcess(image_B, width_B, height_B, format_B, 1) )
		return -1;
	
	PROFILER_END(PROFILER_PREPROCESS);
	
	// process with TRT
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return -1;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);
	
	const uint32_t cellWidth = mInputWidth / mCellResolution;
	const uint32_t cellHeight = mInputWidth / mCellResolution;
	
	const float scale = float(mInputHeight) / float(cellHeight);
	
	const float2 scale_A = make_float2(scale * (float(width_A) / float(mInputWidth)),
								scale * (float(height_A) / float(mInputHeight)));
								
	const float2 scale_B = make_float2(scale * (float(width_B) / float(mInputWidth)),
								scale * (float(height_B) / float(mInputHeight)));
								
	// threshold the confidence matrix
	uint32_t numMatches = 0;
	
	for( uint32_t cx=0; cx < mMaxFeatures; cx++ )
	{
		for( uint32_t cy=0; cy < mMaxFeatures; cy++ )
		{
			const float conf = mOutputs[1].CPU[cx * mMaxFeatures + cy];
			
			if( conf < threshold )
				continue;
			
			const float2 keyA = make_float2((cx % cellWidth) * scale_A.x, int(cx / cellWidth) * scale_A.y);
			const float2 keyB = make_float2((cy % cellWidth) * scale_B.x, int(cy / cellWidth) * scale_B.y);
			
			printf("match %u   %i %i  (%f, %f) -> (%f, %f)\n", numMatches, cx, cy, keyA.x, keyA.y, keyB.x, keyB.y);
			
			if( numMatches == 0 || !sorted )
			{
				features_A[numMatches] = keyA;
				features_B[numMatches] = keyB;
				confidence[numMatches] = conf;
			}
			else
			{
				for( uint32_t n=0; n < numMatches; n++ )
				{
					if( conf > confidence[n] )
					{
						// TODO:  replace these memmoves with a linked/indexed list
						memmove(features_A + n + 1, features_A + n, sizeof(float2) * (numMatches - n));
						memmove(features_B + n + 1, features_B + n, sizeof(float2) * (numMatches - n));
						memmove(confidence + n + 1, confidence + n, sizeof(float) * (numMatches - n));
						
						features_A[n] = keyA;
						features_B[n] = keyB;
						confidence[n] = conf;
						
						break;
					}
					else if( n == (numMatches - 1) )
					{
						features_A[n+1] = keyA;
						features_B[n+1] = keyB;
						confidence[n+1] = conf;
					}
				}
			}
			
			numMatches += 1;
		}
	}

	/*for( uint32_t n=0; n < numMatches; n++ )
	{
		const float x1 = (matches[n].x % cellWidth) * scale_A.x;
		const float y1 = int(matches[n].x / cellWidth) * scale_A.y;
		
		const float x2 = (matches[n].y % cellWidth) * scale_B.x;
		const float y2 = int(matches[n].y / cellWidth) * scale_B.y;
		
		printf("match %u   %i %i  (%f, %f) -> (%f, %f)\n", n, matches[n].x, matches[n].y, x1, y1, x2, y2);
	}*/
	
	printf("cell width = %u\n", cellWidth);
	printf("cell height = %u\n", cellHeight);
	printf("scale = %f\n", scale);
	printf("scale_A = (%f, %f)\n", scale_A.x, scale_A.y);
	printf("scale_B = (%f, %f)\n", scale_B.x, scale_B.y);
	
	PROFILER_END(PROFILER_POSTPROCESS);
	
	return numMatches;
}

					
// DrawFeatures
bool featureNet::DrawFeatures( void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
						 float2* features, uint32_t numFeatures, bool drawText, float scale, const float4& color )
{
	// verify parameters
	if( !input || !output || width == 0 || height == 0 || !features || scale <= 0.0f )
	{
		LogError(LOG_TRT "featureNet::DrawFeatures() called with NULL / invalid parameters\n");
		return false;
	}
	
	if( numFeatures == 0 )
	{
		LogWarning(LOG_TRT "featureNet::DrawFeatures() was called with 0 features, skipping.\n");
		return true;
	}
	
	// draw features
	PROFILER_BEGIN(PROFILER_VISUALIZE);
	
	const float circleSize = width * scale;
	
	for( uint32_t n=0; n < numFeatures; n++ )
	{
		CUDA(cudaDrawCircle(input, output, width, height, format,
						features[n].x, features[n].y, circleSize, color));
	}
	
	if( drawText )
	{
		const float textSize = circleSize * 6;
		const float textOffset = circleSize;
		
		// load font if needed
		if( !mFont )
		{	
			mFont = cudaFont::Create(textSize);
			
			if( !mFont )
			{
				LogError(LOG_TRT "featureNet::DrawFeatures() failed to create font object\n");
				return false;
			}
		}
		
		// TODO use string batching interface to cudaFont
		for( uint32_t n=0; n < numFeatures; n++ )
		{
			char str[256];
			sprintf(str, "%i abc", n);
			
			//printf("drawing text '%s' at %i %i\n", str, int(features[n].x + textOffset), int(features[n].y - textOffset));
			
			mFont->OverlayText(output, format, width, height, str,
						    features[n].x + textOffset,
						    features[n].y - textOffset,
						    color/*, make_float4(0, 0, 0, 100)*/);
		}
	}
		
	PROFILER_END(PROFILER_VISUALIZE);
	
	return true;
}					
	

// DrawFeatures
bool featureNet::DrawFeatures( void* image, uint32_t width, uint32_t height, imageFormat format,
				           float2* features, uint32_t numFeatures, bool drawText, float scale, const float4& color )
{
	return DrawFeatures(image, image, width, height, format, features, numFeatures, drawText, scale, color);
}	

	
/*
// preProcess
bool featureNet::preProcess( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	// verify parameters
	if( !image || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "featureNet::PreProcess( 0x%p, %u, %u ) -> invalid parameters\n", image, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "featureNet::Classify() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                        supported formats are:\n");
		LogError(LOG_TRT "                           * rgb8\n");		
		LogError(LOG_TRT "                           * rgba8\n");		
		LogError(LOG_TRT "                           * rgb32f\n");		
		LogError(LOG_TRT "                           * rgba32f\n");

		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	// input tensor dims are:  3x16x112x112 (CxNxHxW)
	const size_t inputFrameSize = mInputs[0].dims.d[2] * mInputs[0].dims.d[3];
	const size_t inputBatchSize = mInputs[0].dims.d[1] * inputFrameSize;
	
	const uint32_t previousInputBuffer = (mCurrentInputBuffer + 1) % 2;

	// shift the inputs down by one frame
	if( CUDA_FAILED(cudaMemcpy(mInputBuffers[mCurrentInputBuffer],
						  mInputBuffers[previousInputBuffer] + inputFrameSize,
						  mInputs[0].size - (inputFrameSize * sizeof(float)),
						  cudaMemcpyDeviceToDevice)) )
	{
		return false;
	}

	// convert input image into tensor format
	if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, and mean pixel subtraction
		// also apply striding so that the color channels are interleaved across the N frames (CxNxHxW)
		if( CUDA_FAILED(cudaTensorNormMeanRGB(image, format, width, height, 
									   mInputBuffers[mCurrentInputBuffer] + inputFrameSize * mCurrentFrameIndex, 
									   mInputs[0].dims.d[2], mInputs[0].dims.d[3], 
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.4344705882352941f, 0.4050980392156863f, 0.3774901960784314f),
									   make_float3(1.0f, 1.0f, 1.0f), 
									   GetStream(), inputBatchSize)) )
		{
			LogError(LOG_TRT "featureNet::PreProcess() -- cudaTensorNormMeanRGB() failed\n");
			return false;
		}
	}
	
	// update frame counters and pointers
	mInputs[0].CUDA = mInputBuffers[mCurrentInputBuffer];
	
	mCurrentInputBuffer = (mCurrentInputBuffer + 1) % 2;
	mCurrentFrameIndex += 1;
	
	if( mCurrentFrameIndex >= mNumFrames )
		mCurrentFrameIndex = mNumFrames - 1;

	PROFILER_END(PROFILER_PREPROCESS);
	return true;
}


// each component will be in the interval (0, 1) and the sum of all the components is 1
static void softmax( float* x, size_t N )
{
	// subtracting the maximum from each value of the input array ensures that the exponent doesn’t overflow
	float max = -INFINITY;
	
	for( size_t n=0; n < N; n++ )
		if( x[n] > max )
			max = x[n];
		
	// exp(x) / sum(exp(x))
	float sum = 0.0f;
	
	for( size_t n=0; n < N; n++ )
		sum += expf(x[n] - max);
	
	const float constant = max + logf(sum);
	
	for( size_t n=0; n < N; n++ )
		x[n] = expf(x[n] - constant);
}

	
// Classify
int featureNet::Classify( void* image, uint32_t width, uint32_t height, imageFormat format, float* confidence )
{
	// verify parameters
	if( !image || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "featureNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", image, width, height);
		return -1;
	}
	
	// downsample and convert to band-sequential BGR
	if( !preProcess(image, width, height, format) )
	{
		LogError(LOG_TRT "featureNet::Classify() -- tensor pre-processing failed\n");
		return -1;
	}
	
	// process with TRT
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return -1;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// apply softmax (the onnx models are missing this)
	softmax(mOutputs[0].CPU, mNumClasses);
	
	// determine the maximum class
	int classIndex = -1;
	float classMax = -1.0f;
	
	for( size_t n=0; n < mNumClasses; n++ )
	{
		const float value = mOutputs[0].CPU[n];
		
		if( value >= 0.01f )
			LogVerbose("class %04zu - %f  (%s)\n", n, value, mClassDesc[n].c_str());
	
		if( value > classMax )
		{
			classIndex = n;
			classMax   = value;
		}
	}
	
	if( confidence != NULL )
		*confidence = classMax;
	
	//printf("\nmaximum class:  #%i  (%f) (%s)\n", classIndex, classMax, mClassDesc[classIndex].c_str());
	PROFILER_END(PROFILER_POSTPROCESS);	
	return classIndex;
}
*/