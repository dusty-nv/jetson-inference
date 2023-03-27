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
 
#include "detectNet.h"
#include "objectTracker.h"
#include "tensorConvert.h"
#include "modelDownloader.h"

#include "cudaMappedMemory.h"
#include "cudaFont.h"
#include "cudaDraw.h"

#include "commandLine.h"
#include "filesystem.h"
#include "logging.h"


#define OUTPUT_CVG  0	// Caffe has output coverage (confidence) heat map
#define OUTPUT_BBOX 1	// Caffe has separate output layer for bounding box

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count

#define OUTPUT_CONF 0	// ONNX SSD-Mobilenet has confidence as first, bbox second

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"
//#define DEBUG_CLUSTERING


// constructor
detectNet::detectNet( float meanPixel ) : tensorNet()
{
	mTracker   = NULL;
	mMeanPixel = meanPixel;
	mLineWidth = 2.0f;

	mNumClasses  = 0;
	mClassColors = NULL;
	
	mDetectionSets = NULL;
	mDetectionSet  = 0;
	mMaxDetections = 0;
	mOverlayAlpha  = DETECTNET_DEFAULT_ALPHA;
	
	mConfidenceThreshold = DETECTNET_DEFAULT_CONFIDENCE_THRESHOLD;
	mClusteringThreshold = DETECTNET_DEFAULT_CLUSTERING_THRESHOLD;
}


// destructor
detectNet::~detectNet()
{
	SAFE_DELETE(mTracker);
	
	CUDA_FREE_HOST(mDetectionSets);
	CUDA_FREE_HOST(mClassColors);
}


// init
bool detectNet::init( const char* prototxt, const char* model, const char* class_labels, const char* class_colors,
			 	  float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
				  uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	LogInfo("\n");
	LogInfo("detectNet -- loading detection network model from:\n");
	LogInfo("          -- prototxt     %s\n", CHECK_NULL_STR(prototxt));
	LogInfo("          -- model        %s\n", CHECK_NULL_STR(model));
	LogInfo("          -- input_blob   '%s'\n", CHECK_NULL_STR(input_blob));
	LogInfo("          -- output_cvg   '%s'\n", CHECK_NULL_STR(coverage_blob));
	LogInfo("          -- output_bbox  '%s'\n", CHECK_NULL_STR(bbox_blob));
	LogInfo("          -- mean_pixel   %f\n", mMeanPixel);
	LogInfo("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	LogInfo("          -- class_colors %s\n", CHECK_NULL_STR(class_colors));
	LogInfo("          -- threshold    %f\n", threshold);
	LogInfo("          -- batch_size   %u\n\n", maxBatchSize);

	// create list of output names	
	std::vector<std::string> output_blobs;

	if( coverage_blob != NULL )
		output_blobs.push_back(coverage_blob);

	if( bbox_blob != NULL )
		output_blobs.push_back(bbox_blob);
	
	// ONNX SSD models require larger workspace size
	if( modelTypeFromPath(model) == MODEL_ONNX )
	{
		size_t gpuMemFree = 0;
		size_t gpuMemTotal = 0;
		
		CUDA(cudaMemGetInfo(&gpuMemFree, &gpuMemTotal));

		if( gpuMemTotal <= (2048 << 20) )
			mWorkspaceSize = 512 << 20;
		else
			mWorkspaceSize = 2048 << 20;
	}

	// load the model
	if( !LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, 
				  maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "detectNet -- failed to initialize.\n");
		return false;
	}
	
	// allocate detection sets
	if( !allocDetections() )
		return false;

	// load class descriptions
	loadClassInfo(class_labels);
	loadClassColors(class_colors);

	// set the specified threshold
	SetConfidenceThreshold(threshold);

	return true;
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, float mean_pixel, 
						const char* class_labels, float threshold,
						const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	return Create(prototxt, model, mean_pixel, class_labels, NULL, threshold, input_blob,
			    coverage_blob, bbox_blob, maxBatchSize, precision, device, allowGPUFallback);
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, float mean_pixel, 
						const char* class_labels, const char* class_colors, float threshold,
						const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	// check for built-in model string
	if( FindModel(DETECTNET_MODEL_TYPE, model) )
	{
		return Create(model, threshold, maxBatchSize, precision, device, allowGPUFallback);
	}
	else if( fileExtension(model).length() == 0 )
	{
		LogError(LOG_TRT "couldn't find built-in detection model '%s'\n", model);
		return NULL;
	}

	// load custom model
	detectNet* net = new detectNet(mean_pixel);
	
	if( !net )
		return NULL;

	if( !net->init(prototxt, model, class_labels, class_colors, threshold, input_blob, coverage_blob, bbox_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;

	return net;
}


// Create (UFF)
detectNet* detectNet::Create( const char* model, const char* class_labels, float threshold, 
						const char* input, const Dims3& inputDims, 
						const char* output, const char* numDetections,
						uint32_t maxBatchSize, precisionType precision,
				   		deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet();
	
	if( !net )
		return NULL;

	LogInfo("\n");
	LogInfo("detectNet -- loading detection network model from:\n");
	LogInfo("          -- model        %s\n", CHECK_NULL_STR(model));
	LogInfo("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
	LogInfo("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
	LogInfo("          -- output_count '%s'\n", CHECK_NULL_STR(numDetections));
	LogInfo("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	LogInfo("          -- threshold    %f\n", threshold);
	LogInfo("          -- batch_size   %u\n\n", maxBatchSize);
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( output != NULL )
		output_blobs.push_back(output);

	if( numDetections != NULL )
		output_blobs.push_back(numDetections);
	
	// load the model
	if( !net->LoadNetwork(NULL, model, NULL, input, inputDims, output_blobs, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// allocate detection sets
	if( !net->allocDetections() )
		return NULL;

	// load class descriptions
	net->loadClassInfo(class_labels);
	net->loadClassColors(NULL);

	// set the specified threshold
	net->SetConfidenceThreshold(threshold);

	return net;
}


// Create
detectNet* detectNet::Create( const char* network, float threshold, uint32_t maxBatchSize, 
						precisionType precision, deviceType device, bool allowGPUFallback )
{
	nlohmann::json model;
	
	if( !DownloadModel(DETECTNET_MODEL_TYPE, network, model) )
		return NULL;
	
	std::string model_dir = "networks/" + model["dir"].get<std::string>() + "/";
	std::string model_path = model_dir + model["model"].get<std::string>();
	std::string prototxt = JSON_STR(model["prototxt"]);
	std::string labels = JSON_STR(model["labels"]);
	std::string colors = JSON_STR(model["colors"]);
	
	if( prototxt.length() > 0 )
		prototxt = model_dir + prototxt;
	
	if( locateFile(labels).length() == 0 )
		labels = model_dir + labels;
	
	if( locateFile(colors).length() == 0 )
		colors = model_dir + colors;
	
	// get model input/output layers
	std::string input = JSON_STR_DEFAULT(model["input"], DETECTNET_DEFAULT_INPUT);
	std::string output_cvg = DETECTNET_DEFAULT_COVERAGE;
	std::string output_bbox = DETECTNET_DEFAULT_BBOX;
	std::string output_count = "";  // uff
	
	nlohmann::json output = model["output"];
	
	if( output.is_object() )
	{
		if( output["cvg"].is_string() )
			output_cvg = output["cvg"].get<std::string>();
		else if( output["scores"].is_string() )
			output_cvg = output["scores"].get<std::string>();
		
		if( output["bbox"].is_string() )
			output_bbox = output["bbox"].get<std::string>();
		
		if( output["count"].is_string() )
			output_count = output["count"].get<std::string>();
	}
	
	// some older model use the mean_pixel setting
	float mean_pixel = 0.0f;
	
	if( model["mean_pixel"].is_number() )
		mean_pixel = model["mean_pixel"].get<float>();
		
	// UFF models need the input dims parsed
	Dims3 input_dims;
	nlohmann::json dims = model["input_dims"];
	
	if( dims.is_array() && dims.size() == 3 )
	{
		for( uint32_t n=0; n < 3; n++ )
			input_dims.d[n] = dims[n].get<int>();
		
		return Create(model_path.c_str(), labels.c_str(), threshold, input.c_str(), input_dims, 
				    output_bbox.c_str(), output_count.c_str(), maxBatchSize, precision, device, allowGPUFallback);
	}
	
	return Create(prototxt.c_str(), model_path.c_str(), mean_pixel, labels.c_str(), colors.c_str(), threshold,
			    input.c_str(), output_cvg.c_str(), output_bbox.c_str(), maxBatchSize, precision, device, allowGPUFallback);
}


// Create
detectNet* detectNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
detectNet* detectNet::Create( const commandLine& cmdLine )
{
	detectNet* net = NULL;

	// parse command line parameters
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "ssd-mobilenet-v2");

	int maxBatchSize = cmdLine.GetInt("batch_size");
	
	if( maxBatchSize < 1 )
		maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
	
	// confidence used to be called threshold (support both)
	float threshold = cmdLine.GetFloat("threshold");
	
	if( threshold == 0.0f )
	{
		threshold = cmdLine.GetFloat("confidence"); 
		
		if( threshold == 0.0f )
			threshold = DETECTNET_DEFAULT_CONFIDENCE_THRESHOLD;
	}

	// parse the model type
	if( !FindModel(DETECTNET_MODEL_TYPE, modelName) )
	{
		const char* prototxt     = cmdLine.GetString("prototxt");
		const char* input        = cmdLine.GetString("input_blob");
		const char* out_blob     = cmdLine.GetString("output_blob");
		const char* out_cvg      = cmdLine.GetString("output_cvg");
		const char* out_bbox     = cmdLine.GetString("output_bbox");
		const char* class_labels = cmdLine.GetString("class_labels");
		const char* class_colors = cmdLine.GetString("class_colors");
		
		if( !input ) 	
			input = DETECTNET_DEFAULT_INPUT;

		if( !out_blob )
		{
			if( !out_cvg )  out_cvg  = DETECTNET_DEFAULT_COVERAGE;
			if( !out_bbox ) out_bbox = DETECTNET_DEFAULT_BBOX;
		}

		if( !class_labels )
			class_labels = cmdLine.GetString("labels");

		if( !class_colors )
			class_colors = cmdLine.GetString("colors");
		
		float meanPixel = cmdLine.GetFloat("mean_pixel");

		net = detectNet::Create(prototxt, modelName, meanPixel, class_labels, class_colors, threshold, input, 
							out_blob ? NULL : out_cvg, out_blob ? out_blob : out_bbox, maxBatchSize);
	}
	else
	{
		// create detectNet from pretrained model
		net = detectNet::Create(modelName, threshold, maxBatchSize);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set some additional options
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", DETECTNET_DEFAULT_ALPHA));
	net->SetClusteringThreshold(cmdLine.GetFloat("clustering", DETECTNET_DEFAULT_CLUSTERING_THRESHOLD));
	
	// enable tracking if requested
	net->SetTracker(objectTracker::Create(cmdLine));
	
	return net;
}
	

// allocDetections
bool detectNet::allocDetections()
{
	// determine max detections
	if( IsModelType(MODEL_UFF) )	// TODO:  fixme
	{
		LogInfo(LOG_TRT "W = %u  H = %u  C = %u\n", DIMS_W(mOutputs[OUTPUT_UFF].dims), DIMS_H(mOutputs[OUTPUT_UFF].dims), DIMS_C(mOutputs[OUTPUT_UFF].dims));
		mMaxDetections = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		mNumClasses = DIMS_H(mOutputs[OUTPUT_CONF].dims);
		mMaxDetections = DIMS_C(mOutputs[OUTPUT_CONF].dims) /** mNumClasses*/;
		LogInfo(LOG_TRT "detectNet -- number of object classes: %u\n", mNumClasses);
	}	
	else
	{
		mNumClasses = DIMS_C(mOutputs[OUTPUT_CVG].dims);
		mMaxDetections = DIMS_W(mOutputs[OUTPUT_CVG].dims) * DIMS_H(mOutputs[OUTPUT_CVG].dims) * mNumClasses;
		LogInfo(LOG_TRT "detectNet -- number of object classes: %u\n", mNumClasses);
	}

	LogVerbose(LOG_TRT "detectNet -- maximum bounding boxes:   %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets, det_size) )
		return false;
	
	memset(mDetectionSets, 0, det_size);
	return true;
}


// loadClassInfo
bool detectNet::loadClassInfo( const char* filename )
{
	if( !LoadClassLabels(filename, mClassDesc, mClassSynset, mNumClasses) )
		return false;

	if( IsModelType(MODEL_UFF) )
		mNumClasses = mClassDesc.size();

	LogInfo(LOG_TRT "detectNet -- number of object classes:  %u\n", mNumClasses);
	
	if( filename != NULL )
		mClassPath = locateFile(filename);	
	
	return true;
}


// loadClassColors
bool detectNet::loadClassColors( const char* filename )
{
	return LoadClassColors(filename, &mClassColors, mNumClasses, DETECTNET_DEFAULT_ALPHA);
}


// Detect
int detectNet::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	return Detect((void*)input, width, height, IMAGE_RGBA32F, detections, overlay);
}


// Detect
int detectNet::Detect( void* input, uint32_t width, uint32_t height, imageFormat format, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets + mDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;
	
	return Detect(input, width, height, format, det, overlay);
}


// Detect
int detectNet::Detect( float* input, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	return Detect((void*)input, width, height, IMAGE_RGBA32F, detections, overlay);
}


// Detect
int detectNet::Detect( void* input, uint32_t width, uint32_t height, imageFormat format, Detection* detections, uint32_t overlay )
{
	// verify parameters
	if( !input || width == 0 || height == 0 || !detections )
	{
		LogError(LOG_TRT "detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", input, width, height);
		return -1;
	}
	
	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "detectNet::Detect() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "                       supported formats are:\n");
		LogError(LOG_TRT "                          * rgb8\n");		
		LogError(LOG_TRT "                          * rgba8\n");		
		LogError(LOG_TRT "                          * rgb32f\n");		
		LogError(LOG_TRT "                          * rgba32f\n");

		return false;
	}
	
	// apply input pre-processing
	if( !preProcess(input, width, height, format) )
		return -1;
	
	// process model with TensorRT 
	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !ProcessNetwork() )
		return -1;
	
	PROFILER_END(PROFILER_NETWORK);
	
	// post-processing / clustering
	const int numDetections = postProcess(input, width, height, format, detections);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(input, input, width, height, format, detections, numDetections, overlay) )
			LogError(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	//CUDA(cudaDeviceSynchronize());	// BUG is this needed here?

	// return the number of detections
	return numDetections;
}


// preProcess
bool detectNet::preProcess( void* input, uint32_t width, uint32_t height, imageFormat format )
{
	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_UFF) )
	{
		// SSD (TensorFlow / UFF)
		if( CUDA_FAILED(cudaTensorNormBGR(input, format, width, height, 
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float2(-1.0f, 1.0f), GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorNormBGR() failed\n");
			return false;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// SSD (PyTorch / ONNX)
		if( CUDA_FAILED(cudaTensorNormMeanRGB(input, format, width, height,
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.5f, 0.5f, 0.5f),
									   make_float3(0.5f, 0.5f, 0.5f), 
									   GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorNormMeanRGB() failed\n");
			return false;
		}
	}
	else if( IsModelType(MODEL_CAFFE) )
	{
		// DetectNet (Caffe)
		if( CUDA_FAILED(cudaTensorMeanBGR(input, format, width, height,
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float3(mMeanPixel, mMeanPixel, mMeanPixel), 
								    GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorMeanBGR() failed\n");
			return false;
		}
	}
	else if( IsModelType(MODEL_ENGINE) )
	{
		// https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet
		if( CUDA_FAILED(cudaTensorNormRGB(input, format, width, height,
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float2(0.0f, 1.0f), 
								    GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorMeanRGB() failed\n");
			return false;
		}
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	return true;
}


// postProcess
int detectNet::postProcess( void* input, uint32_t width, uint32_t height, imageFormat format, Detection* detections )
{
	PROFILER_BEGIN(PROFILER_POSTPROCESS);
	
	// parse the bounding boxes
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )	
		numDetections = postProcessSSD_UFF(detections, width, height);
	else if( IsModelType(MODEL_ONNX) )
		numDetections = postProcessSSD_ONNX(detections, width, height);
	else if( IsModelType(MODEL_CAFFE) )
		numDetections = postProcessDetectNet(detections, width, height);
	else if( IsModelType(MODEL_ENGINE) )
		numDetections = postProcessDetectNet_v2(detections, width, height);
	else
		return -1;

	// sort the detections by area
	sortDetections(detections, numDetections);
	
	// verify the bounding boxes are within the bounds of the image
	for( int n=0; n < numDetections; n++ )
	{
		if( detections[n].Top < 0 )
			detections[n].Top = 0;
		
		if( detections[n].Left < 0 )
			detections[n].Left = 0;
		
		if( detections[n].Right >= width )
			detections[n].Right = width - 1;
		
		if( detections[n].Bottom >= height )
			detections[n].Bottom = height - 1;
	}
	
	// update tracking
	if( mTracker != NULL && mTracker->IsEnabled() )
		numDetections = mTracker->Process(input, width, height, format, detections, numDetections);
	
	PROFILER_END(PROFILER_POSTPROCESS);	
	return numDetections;
}


// postProcessSSD_UFF
int detectNet::postProcessSSD_UFF( Detection* detections, uint32_t width, uint32_t height )
{
	int numDetections = 0;
	
	const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
	const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

	for( int n=0; n < rawDetections; n++ )
	{
		float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

		if( object_data[2] < mConfidenceThreshold )
			continue;

		detections[numDetections].TrackID   = -1; //numDetections; //(uint32_t)object_data[0];
		detections[numDetections].ClassID    = (uint32_t)object_data[1];
		detections[numDetections].Confidence = object_data[2];
		detections[numDetections].Left       = object_data[3] * width;
		detections[numDetections].Top        = object_data[4] * height;
		detections[numDetections].Right      = object_data[5] * width;
		detections[numDetections].Bottom	  = object_data[6] * height;

		if( detections[numDetections].ClassID >= mNumClasses )
		{
			LogError(LOG_TRT "detectNet::Detect() -- detected object has invalid classID (%u)\n", detections[numDetections].ClassID);
			detections[numDetections].ClassID = 0;
		}

		if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
			continue;

		numDetections += clusterDetections(detections, numDetections);
	}

	return numDetections;
}


// postProcessSSD_ONNX
int detectNet::postProcessSSD_ONNX( Detection* detections, uint32_t width, uint32_t height )
{
	int numDetections = 0;
	
	float* conf = mOutputs[OUTPUT_CONF].CPU;
	float* bbox = mOutputs[OUTPUT_BBOX].CPU;

	const uint32_t numBoxes = DIMS_C(mOutputs[OUTPUT_BBOX].dims);
	const uint32_t numCoord = DIMS_H(mOutputs[OUTPUT_BBOX].dims);

	for( uint32_t n=0; n < numBoxes; n++ )
	{
		uint32_t maxClass = 0;
		float    maxScore = -1000.0f;

		// class #0 in ONNX-SSD is BACKGROUND (ignored)
		for( uint32_t m=1; m < mNumClasses; m++ )	
		{
			const float score = conf[n * mNumClasses + m];

			if( score < mConfidenceThreshold )
				continue;

			if( score > maxScore )
			{
				maxScore = score;
				maxClass = m;
			}
		}

		// check if there was a detection
		if( maxClass <= 0 )
			continue; 

		// populate a new detection entry
		const float* coord = bbox + n * numCoord;

		detections[numDetections].TrackID   = -1; //numDetections;
		detections[numDetections].ClassID    = maxClass;
		detections[numDetections].Confidence = maxScore;
		detections[numDetections].Left       = coord[0] * width;
		detections[numDetections].Top        = coord[1] * height;
		detections[numDetections].Right      = coord[2] * width;
		detections[numDetections].Bottom	  = coord[3] * height;

		if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
			continue;
		
		numDetections += clusterDetections(detections, numDetections);
	}

	return numDetections;
}


// postProcessDetectNet
int detectNet::postProcessDetectNet( Detection* detections, uint32_t width, uint32_t height )
{
	float* net_cvg   = mOutputs[OUTPUT_CVG].CPU;
	float* net_rects = mOutputs[OUTPUT_BBOX].CPU;
	
	const int ow  = DIMS_W(mOutputs[OUTPUT_BBOX].dims);	// number of columns in bbox grid in X dimension
	const int oh  = DIMS_H(mOutputs[OUTPUT_BBOX].dims);	// number of rows in bbox grid in Y dimension
	const int owh = ow * oh;							// total number of bbox in grid
	const int cls = GetNumClasses();					// number of object classes in coverage map
	
	const float cell_width  = /*width*/ GetInputWidth() / ow;
	const float cell_height = /*height*/ GetInputHeight() / oh;
	
	const float scale_x = float(width) / float(GetInputWidth());
	const float scale_y = float(height) / float(GetInputHeight());

#ifdef DEBUG_CLUSTERING	
	LogDebug(LOG_TRT "input width %u height %u\n", GetInputWidth(), GetInputHeight());
	LogDebug(LOG_TRT "cells x %i  y %i\n", ow, oh);
	LogDebug(LOG_TRT "cell width %f  height %f\n", cell_width, cell_height);
	LogDebug(LOG_TRT "scale x %f  y %f\n", scale_x, scale_y);
#endif

	// extract and cluster the raw bounding boxes that meet the coverage threshold
	int numDetections = 0;

	for( uint32_t z=0; z < cls; z++ )	// z = current object class
	{
		for( uint32_t y=0; y < oh; y++ )
		{
			for( uint32_t x=0; x < ow; x++)
			{
				const float coverage = net_cvg[z * owh + y * ow + x];
				
				if( coverage < mConfidenceThreshold )
					continue;

				const float mx = x * cell_width;
				const float my = y * cell_height;
				
				const float x1 = (net_rects[0 * owh + y * ow + x] + mx) * scale_x;	// left
				const float y1 = (net_rects[1 * owh + y * ow + x] + my) * scale_y;	// top
				const float x2 = (net_rects[2 * owh + y * ow + x] + mx) * scale_x;	// right
				const float y2 = (net_rects[3 * owh + y * ow + x] + my) * scale_y;	// bottom 
				
			#ifdef DEBUG_CLUSTERING
				LogDebug(LOG_TRT "rect x=%u y=%u  conf=%f  (%f, %f)  (%f, %f) \n", x, y, coverage, x1, y1, x2, y2);
			#endif		

				// merge with list, checking for overlaps
				bool detectionMerged = false;

				for( uint32_t n=0; n < numDetections; n++ )
				{
					if( detections[n].ClassID == z && detections[n].Expand(x1, y1, x2, y2) )
					{
						detectionMerged = true;
						break;
					}
				}

				// create new entry if the detection wasn't merged with another detection
				if( !detectionMerged )
				{
					detections[numDetections].TrackID   = -1; //numDetections;
					detections[numDetections].ClassID    = z;
					detections[numDetections].Confidence = coverage;
				
					detections[numDetections].Left   = x1;
					detections[numDetections].Top    = y1;
					detections[numDetections].Right  = x2;
					detections[numDetections].Bottom = y2;
				
					numDetections++;
				}
			}
		}
	}
	
	return numDetections;
}


// postProcessDetectNet_v2
int detectNet::postProcessDetectNet_v2( Detection* detections, uint32_t width, uint32_t height )
{
	int numDetections = 0;
	
	float* conf = mOutputs[OUTPUT_CONF].CPU;
	float* bbox = mOutputs[OUTPUT_BBOX].CPU;

	const int cells_x  = DIMS_W(mOutputs[OUTPUT_BBOX].dims);	// number of columns in bbox grid in X dimension
	const int cells_y  = DIMS_H(mOutputs[OUTPUT_BBOX].dims);	// number of rows in bbox grid in Y dimension
	const int numCells = cells_x * cells_y;					// total number of bbox in grid

	const float cell_width  = GetInputWidth() / cells_x;
	const float cell_height = GetInputHeight() / cells_y;
	
	const float scale_x = float(width) / float(GetInputWidth());
	const float scale_y = float(height) / float(GetInputHeight());

	const float bbox_norm = 35.0f;  // https://github.com/NVIDIA-AI-IOT/tao-toolkit-triton-apps/blob/edd383cb2e4c7d18ee95ddbf9fdcf4db7803bb6e/tao_triton/python/postprocessing/detectnet_processor.py#L78
	const float offset = 0.5f;      // https://github.com/NVIDIA-AI-IOT/tao-toolkit-triton-apps/blob/edd383cb2e4c7d18ee95ddbf9fdcf4db7803bb6e/tao_triton/python/postprocessing/detectnet_processor.py#L79
	
#ifdef DEBUG_CLUSTERING	
	LogDebug(LOG_TRT "input width %u height %u\n", GetInputWidth(), GetInputHeight());
	LogDebug(LOG_TRT "cells x %i  y %i\n", cells_x, cells_y);
	LogDebug(LOG_TRT "cell width %f  height %f\n", cell_width, cell_height);
	LogDebug(LOG_TRT "scale x %f  y %f\n", scale_x, scale_y);
#endif

	for( uint32_t c=0; c < mNumClasses; c++ )   // c = current object class
	{
		for( uint32_t y=0; y < cells_y; y++ )
		{
			for( uint32_t x=0; x < cells_x; x++)
			{
				const float confidence = conf[c * numCells + y * cells_x + x];
				
				if( confidence < mConfidenceThreshold )
					continue;

				const float cx = float(x * cell_width + offset) / bbox_norm;
				const float cy = float(y * cell_height + offset) / bbox_norm;
				
				const float x1 = (bbox[(c * 4 + 0) * numCells + y * cells_x + x] - cx) * -bbox_norm * scale_x;
				const float y1 = (bbox[(c * 4 + 1) * numCells + y * cells_x + x] - cy) * -bbox_norm * scale_y;
				const float x2 = (bbox[(c * 4 + 2) * numCells + y * cells_x + x] + cx) *  bbox_norm * scale_x;
				const float y2 = (bbox[(c * 4 + 3) * numCells + y * cells_x + x] + cy) *  bbox_norm * scale_y;
								
			#ifdef DEBUG_CLUSTERING
				LogDebug(LOG_TRT "rect x=%u y=%u  conf=%f  (%f, %f)  (%f, %f) \n", x, y, confidence, x1, y1, x2, y2);
			#endif
				
				detections[numDetections].TrackID   = -1; //numDetections;
				detections[numDetections].ClassID    = c;
				detections[numDetections].Confidence = confidence;
				detections[numDetections].Left       = x1;
				detections[numDetections].Top        = y1;
				detections[numDetections].Right      = x2;
				detections[numDetections].Bottom	  = y2;

				if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
					continue;
		
				numDetections += clusterDetections(detections, numDetections);
			}
		}
	}
	
	return numDetections;
}
	
	
// clusterDetections
int detectNet::clusterDetections( Detection* detections, int n )
{
	if( n == 0 )
		return 1;

	// test each detection to see if it intersects
	for( int m=0; m < n; m++ )
	{
		if( detections[n].Intersects(detections[m], mClusteringThreshold) )	// TODO NMS or different threshold for same classes?
		{
			// if the intersecting detections have different classes, pick the one with highest confidence
			// otherwise if they have the same object class, expand the detection bounding box
		#ifdef CLUSTER_INTERCLASS
			if( detections[n].ClassID != detections[m].ClassID )
			{
				if( detections[n].Confidence > detections[m].Confidence )
				{
					detections[m] = detections[n];

					detections[m].TrackID = -1; //m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;	
				}
				
				return 0; // merged detection
			}
			else
		#else
			if( detections[n].ClassID == detections[m].ClassID )
		#endif
			{
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);

				return 0; // merged detection
			}
		}
	}

	return 1;	// new detection
}


// sortDetections (by area)
void detectNet::sortDetections( Detection* detections, int numDetections )
{
	if( numDetections < 2 )
		return;

	// order by area (descending) or confidence (ascending)
	for( int i=0; i < numDetections-1; i++ )
	{
		for( int j=0; j < numDetections-i-1; j++ )
		{
			if( detections[j].Area() < detections[j+1].Area() ) //if( detections[j].Confidence > detections[j+1].Confidence )
			{
				const Detection det = detections[j];
				detections[j] = detections[j+1];
				detections[j+1] = det;
			}
		}
	}

	// renumber the instance ID's
	//for( int i=0; i < numDetections; i++ )
	//	detections[i].TrackID = i;	
}


// from detectNet.cu
cudaError_t cudaDetectionOverlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections, float4* colors );

// Overlay
bool detectNet::Overlay( void* input, void* output, uint32_t width, uint32_t height, imageFormat format, Detection* detections, uint32_t numDetections, uint32_t flags )
{
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	if( flags == 0 )
	{
		LogError(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_NONE, returning false\n");
		return false;
	}

	// if input and output are different images, copy the input to the output first
	// then overlay the bounding boxes, ect. on top of the output image
	if( input != output )
	{
		if( CUDA_FAILED(cudaMemcpy(output, input, imageFormatSize(format, width, height), cudaMemcpyDeviceToDevice)) )
		{
			LogError(LOG_TRT "detectNet -- Overlay() failed to copy input image to output image\n");
			return false;
		}
	}

	// make sure there are actually detections
	if( numDetections <= 0 )
	{
		PROFILER_END(PROFILER_VISUALIZE);
		return true;
	}

	// bounding box overlay
	if( flags & OVERLAY_BOX )
	{
		if( CUDA_FAILED(cudaDetectionOverlay(input, output, width, height, format, detections, numDetections, mClassColors)) )
			return false;
	}
	
	// bounding box lines
	if( flags & OVERLAY_LINES )
	{
		for( uint32_t n=0; n < numDetections; n++ )
		{
			const Detection* d = detections + n;
			const float4& color = mClassColors[d->ClassID];

			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top, d->Right, d->Top, color, mLineWidth));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Right, d->Top, d->Right, d->Bottom, color, mLineWidth));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Bottom, d->Right, d->Bottom, color, mLineWidth));
			CUDA(cudaDrawLine(input, output, width, height, format, d->Left, d->Top, d->Left, d->Bottom, color, mLineWidth));
		}
	}
			
	// class label overlay
	if( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) || (flags & OVERLAY_TRACKING) )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create(adaptFontSize(width));  // 20.0f
	
			if( !font )
			{
				LogError(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
	#ifdef BATCH_TEXT
		std::vector<std::pair<std::string, int2>> labels;
	#endif 
		for( uint32_t n=0; n < numDetections; n++ )
		{
			const char* className  = GetClassDesc(detections[n].ClassID);
			const float confidence = detections[n].Confidence * 100.0f;
			const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3);
			
			char buffer[256];
			char* str = buffer;
			
			if( flags & OVERLAY_LABEL )
				str += sprintf(str, "%s ", className);
			
			if( flags & OVERLAY_TRACKING && detections[n].TrackID >= 0 )
				str += sprintf(str, "%i ", detections[n].TrackID);
			
			if( flags & OVERLAY_CONFIDENCE )
				str += sprintf(str, "%.1f%%", confidence);

		#ifdef BATCH_TEXT
			labels.push_back(std::pair<std::string, int2>(buffer, position));
		#else
			float4 color = make_float4(255,255,255,255);
		
			if( detections[n].TrackID >= 0 )
				color.w *= 1.0f - (fminf(detections[n].TrackLost, 15.0f) / 15.0f);
			
			font->OverlayText(output, format, width, height, buffer, position.x, position.y, color);
		#endif
		}

	#ifdef BATCH_TEXT
		font->OverlayText(output, format, width, height, labels, make_float4(255,255,255,255));
	#endif
	}
	
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// OverlayFlagsFromStr
uint32_t detectNet::OverlayFlagsFromStr( const char* str_user )
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
		LogError(LOG_TRT "detectNet::OverlayFlagsFromStr() overlay string exceeded max length of %zu characters ('%s')", max_length, str_user);
		return OVERLAY_DEFAULT;
	}
	
	char str[max_length];
	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
		return OVERLAY_DEFAULT;

	// look for the tokens:  "box", "label", "default", and "none"
	uint32_t flags = OVERLAY_NONE;

	while( token != NULL )
	{
		if( strcasecmp(token, "box") == 0 )
			flags |= OVERLAY_BOX;
		else if( strcasecmp(token, "label") == 0 || strcasecmp(token, "labels") == 0 )
			flags |= OVERLAY_LABEL;
		else if( strcasecmp(token, "conf") == 0 || strcasecmp(token, "confidence") == 0 )
			flags |= OVERLAY_CONFIDENCE;
		else if( strcasecmp(token, "track") == 0 || strcasecmp(token, "tracking") == 0 )
			flags |= OVERLAY_TRACKING;
		else if( strcasecmp(token, "line") == 0 || strcasecmp(token, "lines") == 0 )
			flags |= OVERLAY_LINES;
		else if( strcasecmp(token, "default") == 0 )
			flags |= OVERLAY_DEFAULT;

		token = strtok(NULL, delimiters);
	}	

	return flags;
}


// SetOverlayAlpha
void detectNet::SetOverlayAlpha( float alpha )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
		mClassColors[n].w = alpha;
	
	mOverlayAlpha = alpha;
}
