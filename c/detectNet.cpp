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
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaFont.h"

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
	mCoverageThreshold = DETECTNET_DEFAULT_THRESHOLD;
	mMeanPixel         = meanPixel;
	mNumClasses        = 0;

	mClassColors[0]   = NULL; // cpu ptr
	mClassColors[1]   = NULL; // gpu ptr
	
	mDetectionSets[0] = NULL; // cpu ptr
	mDetectionSets[1] = NULL; // gpu ptr
	mDetectionSet     = 0;
	mMaxDetections    = 0;
}


// destructor
detectNet::~detectNet()
{
	if( mDetectionSets != NULL )
	{
		CUDA(cudaFreeHost(mDetectionSets[0]));
		
		mDetectionSets[0] = NULL;
		mDetectionSets[1] = NULL;
	}
	
	if( mClassColors != NULL )
	{
		CUDA(cudaFreeHost(mClassColors[0]));
		
		mClassColors[0] = NULL;
		mClassColors[1] = NULL;
	}
}


// init
bool detectNet::init( const char* prototxt, const char* model, const char* mean_binary, const char* class_labels, 
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
	LogInfo("          -- mean_binary  %s\n", CHECK_NULL_STR(mean_binary));
	LogInfo("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
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
	if( !LoadNetwork(prototxt, model, mean_binary, input_blob, output_blobs, 
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
	
	// set default class colors
	if( !defaultColors() )
		return false;

	// set the specified threshold
	SetThreshold(threshold);

	return true;
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, float mean_pixel, const char* class_labels,
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet(mean_pixel);
	
	if( !net )
		return NULL;

	if( !net->init(prototxt, model, NULL, class_labels, threshold, input_blob, coverage_blob, bbox_blob,
				maxBatchSize, precision, device, allowGPUFallback) )
		return NULL;

	return net;
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, const char* mean_binary, const char* class_labels, 
						float threshold, const char* input_blob, const char* coverage_blob, const char* bbox_blob, 
						uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	detectNet* net = new detectNet();
	
	if( !net )
		return NULL;

	if( !net->init(prototxt, model, mean_binary, class_labels, threshold, input_blob, coverage_blob, bbox_blob,
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
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	// set the specified threshold
	net->SetThreshold(threshold);

	return net;
}


// Create
detectNet* detectNet::Create( NetworkType networkType, float threshold, uint32_t maxBatchSize, 
						precisionType precision, deviceType device, bool allowGPUFallback )
{
#if 1
	if( networkType == PEDNET_MULTI )
		return Create("networks/multiped-500/deploy.prototxt", "networks/multiped-500/snapshot_iter_178000.caffemodel", 117.0f, "networks/multiped-500/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == FACENET )
		return Create("networks/facenet-120/deploy.prototxt", "networks/facenet-120/snapshot_iter_24000.caffemodel", 0.0f, "networks/facenet-120/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == PEDNET )
		return Create("networks/ped-100/deploy.prototxt", "networks/ped-100/snapshot_iter_70800.caffemodel", 0.0f, "networks/ped-100/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_AIRPLANE )
		return Create("networks/DetectNet-COCO-Airplane/deploy.prototxt", "networks/DetectNet-COCO-Airplane/snapshot_iter_22500.caffemodel", 0.0f, "networks/DetectNet-COCO-Airplane/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_BOTTLE )
		return Create("networks/DetectNet-COCO-Bottle/deploy.prototxt", "networks/DetectNet-COCO-Bottle/snapshot_iter_59700.caffemodel", 0.0f, "networks/DetectNet-COCO-Bottle/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_CHAIR )
		return Create("networks/DetectNet-COCO-Chair/deploy.prototxt", "networks/DetectNet-COCO-Chair/snapshot_iter_89500.caffemodel", 0.0f, "networks/DetectNet-COCO-Chair/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_DOG )
		return Create("networks/DetectNet-COCO-Dog/deploy.prototxt", "networks/DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel", 0.0f, "networks/DetectNet-COCO-Dog/class_labels.txt", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
#if NV_TENSORRT_MAJOR > 4
	else if( networkType == SSD_INCEPTION_V2 )
		return Create("networks/SSD-Inception-v2/ssd_inception_v2_coco.uff", "networks/SSD-Inception-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V1 )
		return Create("networks/SSD-Mobilenet-v1/ssd_mobilenet_v1_coco.uff", "networks/SSD-Mobilenet-v1/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "Postprocessor", "Postprocessor_1", maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == SSD_MOBILENET_V2 )
		return Create("networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff", "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt", threshold, "Input", Dims3(3,300,300), "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
#endif
	else
		return NULL;
#else
	if( networkType == PEDNET_MULTI )
		return Create("networks/multiped-500/deploy.prototxt", "networks/multiped-500/snapshot_iter_178000.caffemodel", "networks/multiped-500/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == FACENET )
		return Create("networks/facenet-120/deploy.prototxt", "networks/facenet-120/snapshot_iter_24000.caffemodel", NULL, threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == PEDNET )
		return Create("networks/ped-100/deploy.prototxt", "networks/ped-100/snapshot_iter_70800.caffemodel", "networks/ped-100/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_AIRPLANE )
		return Create("networks/DetectNet-COCO-Airplane/deploy.prototxt", "networks/DetectNet-COCO-Airplane/snapshot_iter_22500.caffemodel", "networks/DetectNet-COCO-Airplane/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_BOTTLE )
		return Create("networks/DetectNet-COCO-Bottle/deploy.prototxt", "networks/DetectNet-COCO-Bottle/snapshot_iter_59700.caffemodel", "networks/DetectNet-COCO-Bottle/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_CHAIR )
		return Create("networks/DetectNet-COCO-Chair/deploy.prototxt", "networks/DetectNet-COCO-Chair/snapshot_iter_89500.caffemodel", "networks/DetectNet-COCO-Chair/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == COCO_DOG )
		return Create("networks/DetectNet-COCO-Dog/deploy.prototxt", "networks/DetectNet-COCO-Dog/snapshot_iter_38600.caffemodel", "networks/DetectNet-COCO-Dog/mean.binaryproto", threshold, DETECTNET_DEFAULT_INPUT, DETECTNET_DEFAULT_COVERAGE, DETECTNET_DEFAULT_BBOX, maxBatchSize, precision, device, allowGPUFallback );
	else 
		return NULL;
#endif
}


// NetworkTypeFromStr
detectNet::NetworkType detectNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return detectNet::CUSTOM;

	detectNet::NetworkType type = detectNet::PEDNET;

	if( strcasecmp(modelName, "multiped") == 0 || strcasecmp(modelName, "multiped-500") == 0 )
		type = detectNet::PEDNET_MULTI;
	else if( strcasecmp(modelName, "pednet") == 0 || strcasecmp(modelName, "ped-100") == 0 )
		type = detectNet::PEDNET;
	else if( strcasecmp(modelName, "facenet") == 0 || strcasecmp(modelName, "facenet-120") == 0 || strcasecmp(modelName, "face-120") == 0 )
		type = detectNet::FACENET;
	else if( strcasecmp(modelName, "coco-airplane") == 0 || strcasecmp(modelName, "airplane") == 0 )
		type = detectNet::COCO_AIRPLANE;
	else if( strcasecmp(modelName, "coco-bottle") == 0 || strcasecmp(modelName, "bottle") == 0 )
		type = detectNet::COCO_BOTTLE;
	else if( strcasecmp(modelName, "coco-chair") == 0 || strcasecmp(modelName, "chair") == 0 )
		type = detectNet::COCO_CHAIR;
	else if( strcasecmp(modelName, "coco-dog") == 0 || strcasecmp(modelName, "dog") == 0 )
		type = detectNet::COCO_DOG;
#if NV_TENSORRT_MAJOR > 4
	else if( strcasecmp(modelName, "ssd-inception") == 0 || strcasecmp(modelName, "ssd-inception-v2") == 0 || strcasecmp(modelName, "coco-ssd-inception") == 0 || strcasecmp(modelName, "coco-ssd-inception-v2") == 0)
		type = detectNet::SSD_INCEPTION_V2;
	else if( strcasecmp(modelName, "ssd-mobilenet-v1") == 0 || strcasecmp(modelName, "coco-ssd-mobilenet-v1") == 0)
		type = detectNet::SSD_MOBILENET_V1;
	else if( strcasecmp(modelName, "ssd-mobilenet-v2") == 0 || strcasecmp(modelName, "coco-ssd-mobilenet-v2") == 0 || strcasecmp(modelName, "ssd-mobilenet") == 0 )
		type = detectNet::SSD_MOBILENET_V2;
#endif
	else
		type = detectNet::CUSTOM;

	return type;
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

	float threshold = cmdLine.GetFloat("threshold");
	
	if( threshold == 0.0f )
		threshold = DETECTNET_DEFAULT_THRESHOLD;
	
	int maxBatchSize = cmdLine.GetInt("batch_size");
	
	if( maxBatchSize < 1 )
		maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

	// parse the model type
	const detectNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == detectNet::CUSTOM )
	{
		const char* prototxt     = cmdLine.GetString("prototxt");
		const char* input        = cmdLine.GetString("input_blob");
		const char* out_blob     = cmdLine.GetString("output_blob");
		const char* out_cvg      = cmdLine.GetString("output_cvg");
		const char* out_bbox     = cmdLine.GetString("output_bbox");
		const char* class_labels = cmdLine.GetString("class_labels");

		if( !input ) 	
			input = DETECTNET_DEFAULT_INPUT;

		if( !out_blob )
		{
			if( !out_cvg )  out_cvg  = DETECTNET_DEFAULT_COVERAGE;
			if( !out_bbox ) out_bbox = DETECTNET_DEFAULT_BBOX;
		}

		if( !class_labels )
			class_labels = cmdLine.GetString("labels");

		float meanPixel = cmdLine.GetFloat("mean_pixel");

		net = detectNet::Create(prototxt, modelName, meanPixel, class_labels, threshold, input, 
							out_blob ? NULL : out_cvg, out_blob ? out_blob : out_bbox, maxBatchSize);
	}
	else
	{
		// create detectNet from pretrained model
		net = detectNet::Create(type, threshold, maxBatchSize);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set overlay alpha value
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", DETECTNET_DEFAULT_ALPHA));

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
		LogInfo(LOG_TRT "detectNet -- number object classes:  %u\n", mNumClasses);
	}	
	else
	{
		mNumClasses = DIMS_C(mOutputs[OUTPUT_CVG].dims);
		mMaxDetections = DIMS_W(mOutputs[OUTPUT_CVG].dims) * DIMS_H(mOutputs[OUTPUT_CVG].dims) /** DIMS_C(mOutputs[OUTPUT_CVG].dims)*/ * mNumClasses;
		LogInfo(LOG_TRT "detectNet -- number object classes:   %u\n", mNumClasses);
	}

	LogVerbose(LOG_TRT "detectNet -- maximum bounding boxes:  %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;
}


// GenerateColor
void detectNet::GenerateColor( uint32_t classID, uint8_t* rgb )
{
	if( !rgb )
		return;

	// the first color is black, skip that one
	classID += 1;

	// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
	#define bitget(byteval, idx)	((byteval & (1 << idx)) != 0)
	
	int r = 0;
	int g = 0;
	int b = 0;
	int c = classID;

	for( int j=0; j < 8; j++ )
	{
		r = r | (bitget(c, 0) << 7 - j);
		g = g | (bitget(c, 1) << 7 - j);
		b = b | (bitget(c, 2) << 7 - j);
		c = c >> 3;
	}

	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}


// defaultColors
bool detectNet::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	
	if( !cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) )
		return false;

	// if there are a large number of classes (MS COCO)
	// programatically generate the class color map
	if( mModelType != MODEL_CAFFE /*numClasses > 10*/ )
	{
		for( int i=0; i < numClasses; i++ )
		{
			uint8_t rgb[] = {0,0,0};
			GenerateColor(i, rgb);

			mClassColors[0][i*4+0] = rgb[0];
			mClassColors[0][i*4+1] = rgb[1];
			mClassColors[0][i*4+2] = rgb[2];
			mClassColors[0][i*4+3] = DETECTNET_DEFAULT_ALPHA; 

			//printf(LOG_TRT "color %02i  %3i %3i %3i %3i\n", i, (int)r, (int)g, (int)b, (int)alpha);
		}
	}
	else
	{
		// blue colors, except class 1 is green
		for( uint32_t n=0; n < numClasses; n++ )
		{
			if( n != 1 )
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 200.0f;	// g
				mClassColors[0][n*4+2] = 255.0f;	// b
				mClassColors[0][n*4+3] = DETECTNET_DEFAULT_ALPHA;	// a
			}
			else
			{
				mClassColors[0][n*4+0] = 0.0f;	// r
				mClassColors[0][n*4+1] = 255.0f;	// g
				mClassColors[0][n*4+2] = 175.0f;	// b
				mClassColors[0][n*4+3] = 75.0f;	// a
			}
		}
	}

	return true;
}


// LoadClassInfo
bool detectNet::LoadClassInfo( const char* filename, std::vector<std::string>& descriptions, std::vector<std::string>& synsets, int expectedClasses )
{
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "detectNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "detectNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	descriptions.clear();
	synsets.clear();

	// read class descriptions
	char str[512];
	uint32_t customClasses = 0;

	while( fgets(str, 512, f) != NULL )
	{
		const int syn = 9;  // length of synset prefix (in characters)
		const int len = strlen(str);
		
		if( len > syn && str[0] == 'n' && str[syn] == ' ' )
		{
			str[syn]   = 0;
			str[len-1] = 0;
	
			const std::string a = str;
			const std::string b = (str + syn + 1);
	
			//printf("a=%s b=%s\n", a.c_str(), b.c_str());

			synsets.push_back(a);
			descriptions.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", customClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			customClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			synsets.push_back(a);
			descriptions.push_back(str);
		}
	}
	
	fclose(f);
	
	LogVerbose(LOG_TRT "detectNet -- loaded %zu class info entries\n", synsets.size());
	
	const int numLoaded = descriptions.size();

	if( numLoaded == 0 )
		return false;

	if( expectedClasses > 0 )
	{
		if( numLoaded != expectedClasses )
			LogError(LOG_TRT "detectNet -- didn't load expected number of class descriptions  (%i of %i)\n", numLoaded, expectedClasses);

		if( numLoaded < expectedClasses )
		{
			LogWarning(LOG_TRT "detectNet -- filling in remaining %i class descriptions with default labels\n", (expectedClasses - numLoaded));
 
			for( int n=numLoaded; n < expectedClasses; n++ )
			{
				char synset[10];
				sprintf(synset, "n%08i", n);

				char desc[64];
				sprintf(desc, "Class #%i", n);

				synsets.push_back(synset);
				descriptions.push_back(desc);
			}
		}
	}

	return true;
}


// LoadClassInfo
bool detectNet::LoadClassInfo( const char* filename, std::vector<std::string>& descriptions, int expectedClasses )
{
	std::vector<std::string> synsets;
	return LoadClassInfo(filename, descriptions, synsets, expectedClasses);
}


// loadClassInfo
bool detectNet::loadClassInfo( const char* filename )
{
	if( !LoadClassInfo(filename, mClassDesc, mClassSynset, mNumClasses) )
		return false;

	if( IsModelType(MODEL_UFF) )
		mNumClasses = mClassDesc.size();

	LogInfo(LOG_TRT "detectNet -- number of object classes:  %u\n", mNumClasses);
	mClassPath = locateFile(filename);	
	return true;
}


// Detect
int detectNet::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	return Detect((void*)input, width, height, IMAGE_RGBA32F, detections, overlay);
}


// Detect
int detectNet::Detect( void* input, uint32_t width, uint32_t height, imageFormat format, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets[0] + mDetectionSet * GetMaxDetections();

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

		return -1;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_UFF) )
	{
		if( CUDA_FAILED(cudaTensorNormBGR(input, format, width, height, 
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float2(-1.0f, 1.0f), GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorNormBGR() failed\n");
			return -1;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaTensorNormMeanRGB(input, format, width, height,
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.5f, 0.5f, 0.5f),
									   make_float3(0.5f, 0.5f, 0.5f), 
									   GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorNormMeanRGB() failed\n");
			return -1;
		}
	}
	else
	{
		if( CUDA_FAILED(cudaTensorMeanBGR(input, format, width, height,
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float3(mMeanPixel, mMeanPixel, mMeanPixel), 
								    GetStream())) )
		{
			LogError(LOG_TRT "detectNet::Detect() -- cudaTensorMeanBGR() failed\n");
			return -1;
		}
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	// process with TensorRT
	/*void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}*/

	if( !ProcessNetwork() )
		return -1;
	
	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )
	{		
		const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

#ifdef DEBUG_CLUSTERING	
		LogDebug(LOG_TRT "detectNet::Detect() -- %i unfiltered detections\n", rawDetections);
#endif

		// filter the raw detections by thresholding the confidence
		for( int n=0; n < rawDetections; n++ )
		{
			float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

			if( object_data[2] < mCoverageThreshold )
				continue;

			detections[numDetections].Instance   = numDetections; //(uint32_t)object_data[0];
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

		// sort the detections by confidence value
		sortDetections(detections, numDetections);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
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

				if( score < mCoverageThreshold )
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

			detections[numDetections].Instance   = numDetections;
			detections[numDetections].ClassID    = maxClass;
			detections[numDetections].Confidence = maxScore;
			detections[numDetections].Left       = coord[0] * width;
			detections[numDetections].Top        = coord[1] * height;
			detections[numDetections].Right      = coord[2] * width;
			detections[numDetections].Bottom	  = coord[3] * height;

			numDetections += clusterDetections(detections, numDetections);
		}

		// sort the detections by confidence value
		sortDetections(detections, numDetections);
	}
	else
	{
		// cluster detections
		numDetections = clusterDetections(detections, width, height);
	}

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
	
	PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(input, input, width, height, format, detections, numDetections, overlay) )
			LogError(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}


// clusterDetections (UFF/ONNX)
int detectNet::clusterDetections( Detection* detections, int n, float threshold )
{
	if( n == 0 )
		return 1;

	// test each detection to see if it intersects
	for( int m=0; m < n; m++ )
	{
		if( detections[n].Intersects(detections[m], threshold) )	// TODO NMS or different threshold for same classes?
		{
			// if the intersecting detections have different classes, pick the one with highest confidence
			// otherwise if they have the same object class, expand the detection bounding box
			if( detections[n].ClassID != detections[m].ClassID )
			{
				if( detections[n].Confidence > detections[m].Confidence )
				{
					detections[m] = detections[n];

					detections[m].Instance = m;
					detections[m].ClassID = detections[n].ClassID;
					detections[m].Confidence = detections[n].Confidence;					
				}
			}
			else
			{
				detections[m].Expand(detections[n]);
				detections[m].Confidence = fmaxf(detections[n].Confidence, detections[m].Confidence);
			}

			return 0; // merged detection
		}
	}

	return 1;	// new detection
}


// clusterDetections (caffe)
int detectNet::clusterDetections( Detection* detections, uint32_t width, uint32_t height )
{
	// cluster detection bboxes
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
	LogDebug(LOG_TRT "input width %i height %i\n", (int)DIMS_W(mInputDims), (int)DIMS_H(mInputDims));
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
				
				if( coverage > mCoverageThreshold )
				{
					const float mx = x * cell_width;
					const float my = y * cell_height;
					
					const float x1 = (net_rects[0 * owh + y * ow + x] + mx) * scale_x;	// left
					const float y1 = (net_rects[1 * owh + y * ow + x] + my) * scale_y;	// top
					const float x2 = (net_rects[2 * owh + y * ow + x] + mx) * scale_x;	// right
					const float y2 = (net_rects[3 * owh + y * ow + x] + my) * scale_y;	// bottom 
					
				#ifdef DEBUG_CLUSTERING
					LogDebug(LOG_TRT "rect x=%u y=%u  cvg=%f  %f %f   %f %f \n", x, y, coverage, x1, x2, y1, y2);
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
						detections[numDetections].Instance   = numDetections;
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
	}
	
	return numDetections;
}


// sortDetections (UFF)
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
	for( int i=0; i < numDetections; i++ )
		detections[i].Instance = i;	
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
		if( CUDA_FAILED(cudaDetectionOverlay(input, output, width, height, format, detections, numDetections, (float4*)mClassColors[1])) )
			return false;
	}

	// class label overlay
	if( (flags & OVERLAY_LABEL) || (flags & OVERLAY_CONFIDENCE) )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create(adaptFontSize(width));
	
			if( !font )
			{
				LogError(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector< std::pair< std::string, int2 > > labels;

		for( uint32_t n=0; n < numDetections; n++ )
		{
			const char* className  = GetClassDesc(detections[n].ClassID);
			const float confidence = detections[n].Confidence * 100.0f;
			const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3);
			
			if( flags & OVERLAY_CONFIDENCE )
			{
				char str[256];

				if( (flags & OVERLAY_LABEL) && (flags & OVERLAY_CONFIDENCE) )
					sprintf(str, "%s %.1f%%", className, confidence);
				else
					sprintf(str, "%.1f%%", confidence);

				labels.push_back(std::pair<std::string, int2>(str, position));
			}
			else
			{
				// overlay label only
				labels.push_back(std::pair<std::string, int2>(className, position));
			}
		}

		font->OverlayText(output, format, width, height, labels, make_float4(255,255,255,255));
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
		else if( strcasecmp(token, "default") == 0 )
			flags |= OVERLAY_DEFAULT;

		token = strtok(NULL, delimiters);
	}	

	return flags;
}


// SetClassColor
void detectNet::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[0][i+0] = r;
	mClassColors[0][i+1] = g;
	mClassColors[0][i+2] = b;
	mClassColors[0][i+3] = a;
}


// SetOverlayAlpha
void detectNet::SetOverlayAlpha( float alpha )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
		mClassColors[0][n*4+3] = alpha;
}
