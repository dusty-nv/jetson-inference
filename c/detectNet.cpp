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
#include "imageNet.cuh"

#include "cudaMappedMemory.h"
#include "cudaFont.h"

#include "commandLine.h"
#include "filesystem.h"


#define OUTPUT_CVG  0	// Caffe has output coverage (confidence) heat map
#define OUTPUT_BBOX 1	// Caffe has separate output layer for bounding box

#define OUTPUT_UFF  0	// UFF has primary output containing detection results
#define OUTPUT_NUM	1	// UFF has secondary output containing one detection count

#define CHECK_NULL_STR(x)	(x != NULL) ? x : "NULL"
//#define DEBUG_CLUSTERING


// constructor
detectNet::detectNet( float meanPixel ) : tensorNet()
{
	mCoverageThreshold = DETECTNET_DEFAULT_THRESHOLD;
	mMeanPixel         = meanPixel;
	mCustomClasses     = 0;
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
	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
	printf("          -- prototxt     %s\n", CHECK_NULL_STR(prototxt));
	printf("          -- model        %s\n", CHECK_NULL_STR(model));
	printf("          -- input_blob   '%s'\n", CHECK_NULL_STR(input_blob));
	printf("          -- output_cvg   '%s'\n", CHECK_NULL_STR(coverage_blob));
	printf("          -- output_bbox  '%s'\n", CHECK_NULL_STR(bbox_blob));
	printf("          -- mean_pixel   %f\n", mMeanPixel);
	printf("          -- mean_binary  %s\n", CHECK_NULL_STR(mean_binary));
	printf("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);

	//net->EnableDebug();
	
	// create list of output names	
	std::vector<std::string> output_blobs;

	if( coverage_blob != NULL )
		output_blobs.push_back(coverage_blob);

	if( bbox_blob != NULL )
		output_blobs.push_back(bbox_blob);
	
	// load the model
	if( !LoadNetwork(prototxt, model, mean_binary, input_blob, output_blobs, 
				  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("detectNet -- failed to initialize.\n");
		return false;
	}
	
	// allocate detection sets
	if( !allocDetections() )
		return false;

	// load class descriptions
	loadClassDesc(class_labels);
	defaultClassDesc();
	
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

	printf("\n");
	printf("detectNet -- loading detection network model from:\n");
	printf("          -- model        %s\n", CHECK_NULL_STR(model));
	printf("          -- input_blob   '%s'\n", CHECK_NULL_STR(input));
	printf("          -- output_blob  '%s'\n", CHECK_NULL_STR(output));
	printf("          -- output_count '%s'\n", CHECK_NULL_STR(numDetections));
	printf("          -- class_labels %s\n", CHECK_NULL_STR(class_labels));
	printf("          -- threshold    %f\n", threshold);
	printf("          -- batch_size   %u\n\n", maxBatchSize);
	
	//net->EnableDebug();
	
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
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	// allocate detection sets
	if( !net->allocDetections() )
		return NULL;

	// load class descriptions
	net->loadClassDesc(class_labels);
	net->defaultClassDesc();
	
	// set default class colors
	if( !net->defaultColors() )
		return NULL;

	// set the specified threshold
	net->SetThreshold(threshold);

	return net;
}


// allocDetections
bool detectNet::allocDetections()
{
	// determine max detections
	if( IsModelType(MODEL_UFF) )	// TODO:  fixme
	{
		printf("W = %u  H = %u  C = %u\n", DIMS_W(mOutputs[OUTPUT_UFF].dims), DIMS_H(mOutputs[OUTPUT_UFF].dims), DIMS_C(mOutputs[OUTPUT_UFF].dims));
		mMaxDetections = DIMS_H(mOutputs[OUTPUT_UFF].dims) * DIMS_C(mOutputs[OUTPUT_UFF].dims);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		mMaxDetections = 1;
		mNumClasses = 1;
		printf("detectNet -- using ONNX model\n");
	}	
	else
	{
		mNumClasses = DIMS_C(mOutputs[OUTPUT_CVG].dims);
		mMaxDetections = DIMS_W(mOutputs[OUTPUT_CVG].dims) * DIMS_H(mOutputs[OUTPUT_CVG].dims) /** DIMS_C(mOutputs[OUTPUT_CVG].dims)*/ * mNumClasses;
		printf("detectNet -- number object classes:   %u\n", mNumClasses);
	}

	printf("detectNet -- maximum bounding boxes:  %u\n", mMaxDetections);

	// allocate array to store detection results
	const size_t det_size = sizeof(Detection) * mNumDetectionSets * mMaxDetections;
	
	if( !cudaAllocMapped((void**)&mDetectionSets[0], (void**)&mDetectionSets[1], det_size) )
		return false;
	
	memset(mDetectionSets[0], 0, det_size);
	return true;
}

	
// defaultColors
bool detectNet::defaultColors()
{
	const uint32_t numClasses = GetNumClasses();
	
	if( !cudaAllocMapped((void**)&mClassColors[0], (void**)&mClassColors[1], numClasses * sizeof(float4)) )
		return false;
	
	for( uint32_t n=0; n < numClasses; n++ )
	{
		if( n != 1 )
		{
			mClassColors[0][n*4+0] = 0.0f;	// r
			mClassColors[0][n*4+1] = 200.0f;	// g
			mClassColors[0][n*4+2] = 255.0f;	// b
			mClassColors[0][n*4+3] = 100.0f;	// a
		}
		else
		{
			mClassColors[0][n*4+0] = 0.0f;	// r
			mClassColors[0][n*4+1] = 255.0f;	// g
			mClassColors[0][n*4+2] = 175.0f;	// b
			mClassColors[0][n*4+3] = 75.0f;	// a
		}
	}
	
	return true;
}


// defaultClassDesc
void detectNet::defaultClassDesc()
{
	const uint32_t numClasses = GetNumClasses();
	const int syn = 9;  // length of synset prefix (in characters)
	
	// assign defaults to classes that have no info
	for( uint32_t n=mClassDesc.size(); n < numClasses; n++ )
	{
		char syn_str[10];
		sprintf(syn_str, "n%08u", mCustomClasses);

		char desc_str[16];
		sprintf(desc_str, "class #%u", mCustomClasses);

		mClassSynset.push_back(syn_str);
		mClassDesc.push_back(desc_str);

		mCustomClasses++;
	}
}


// loadClassDesc
bool detectNet::loadClassDesc( const char* filename )
{
	//printf("detectNet -- model has %u object classes\n", GetNumClasses());

	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf("detectNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("detectNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class descriptions
	char str[512];

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

			mClassSynset.push_back(a);
			mClassDesc.push_back(b);
		}
		else if( len > 0 )	// no 9-character synset prefix (i.e. from DIGITS snapshot)
		{
			char a[10];
			sprintf(a, "n%08u", mCustomClasses);

			//printf("a=%s b=%s (custom non-synset)\n", a, str);
			mCustomClasses++;

			if( str[len-1] == '\n' )
				str[len-1] = 0;

			mClassSynset.push_back(a);
			mClassDesc.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("detectNet -- loaded %zu class info entries\n", mClassDesc.size());
	
	//for( size_t n=0; n < mClassDesc.size(); n++ )
		//printf("          -- %s '%s'\n", mClassSynset[n].c_str(), mClassDesc[n].c_str());

	if( mClassDesc.size() == 0 )
		return false;

	if( IsModelType(MODEL_UFF) )
		mNumClasses = mClassDesc.size();

	printf("detectNet -- number of object classes:  %u\n", mNumClasses);
	mClassPath = path;	
	return true;
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
	else if( strcasecmp(modelName, "ssd-mobilenet-v2") == 0 || strcasecmp(modelName, "coco-ssd-mobilenet-v2") == 0)
		type = detectNet::SSD_MOBILENET_V2;
#endif
	else
		type = detectNet::CUSTOM;

	return type;
}


// Create
detectNet* detectNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "pednet");

	/*if( !modelName )
	{
		if( argc == 2 )
			modelName = argv[1];
		else if( argc == 4 )
			modelName = argv[3];
		else
			modelName = "pednet";
	}*/

	float threshold = cmdLine.GetFloat("threshold");
	
	if( threshold == 0.0f )
		threshold = DETECTNET_DEFAULT_THRESHOLD;
	
	int maxBatchSize = cmdLine.GetInt("batch_size");
	
	if( maxBatchSize < 1 )
		maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

	//if( argc > 3 )
	//	modelName = argv[3];	

	const detectNet::NetworkType type = NetworkTypeFromStr(modelName); /*detectNet::PEDNET_MULTI;

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
	else*/
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

		float meanPixel = cmdLine.GetFloat("mean_pixel");

		return detectNet::Create(prototxt, modelName, meanPixel, class_labels, threshold, input, 
							out_blob ? NULL : out_cvg, out_blob ? out_blob : out_bbox, maxBatchSize);
	}

	// create segnet from pretrained model
	return detectNet::Create(type, threshold, maxBatchSize);
}
	

#if 0
inline static bool rectOverlap(const float6& r1, const float6& r2)
{
    return ! ( r2.x > r1.z  
        || r2.z < r1.x
        || r2.y > r1.w
        || r2.w < r1.y
        );
}
#endif


// Detect
int detectNet::Detect( float* input, uint32_t width, uint32_t height, Detection** detections, uint32_t overlay )
{
	Detection* det = mDetectionSets[0] + mDetectionSet * GetMaxDetections();

	if( detections != NULL )
		*detections = det;

	mDetectionSet++;

	if( mDetectionSet >= mNumDetectionSets )
		mDetectionSet = 0;
	
	return Detect(input, width, height, det, overlay);
}

	
// Detect
int detectNet::Detect( float* rgba, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	if( !rgba || width == 0 || height == 0 || !detections )
	{
		printf(LOG_TRT "detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_UFF) )
	{
		if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float2(-1.0f, 1.0f), GetStream())) )
		{
			printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetNorm() failed\n");
			return -1;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.485f, 0.456f, 0.406f),
										   make_float3(0.229f, 0.224f, 0.225f), 
										   GetStream())) )
		{
			printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		if( mMeanPixel != 0.0f )
		{
			if( CUDA_FAILED(cudaPreImageNetMeanBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float3(mMeanPixel, mMeanPixel, mMeanPixel), GetStream())) )
			{
				printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNetMean() failed\n");
				return -1;
			}
		}
		else
		{
			if( CUDA_FAILED(cudaPreImageNetBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
			{
				printf(LOG_TRT "detectNet::Detect() -- cudaPreImageNet() failed\n");
				return -1;
			}
		}
	}
	
	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_TRT "detectNet::Detect() -- failed to execute TensorRT context\n");
		return -1;
	}
	
	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// post-processing / clustering
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )
	{		
		const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

#ifdef DEBUG_CLUSTERING	
		printf(LOG_TRT "detectNet::Detect() -- %i unfiltered detections\n", rawDetections);
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
				printf(LOG_TRT "detectNet::Detect() -- detected object has invalid classID (%u)\n", detections[numDetections].ClassID);
				detections[numDetections].ClassID = 0;
			}

			if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
				continue;

			numDetections++;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		float* coord = mOutputs[0].CPU;

		coord[0] = ((coord[0] + 1.0f) * 0.5f) * float(width);
		coord[1] = ((coord[1] + 1.0f) * 0.5f) * float(height);
		coord[2] = ((coord[2] + 1.0f) * 0.5f) * float(width);
		coord[3] = ((coord[3] + 1.0f) * 0.5f) * float(height);

		printf(LOG_TRT "detectNet::Detect() -- ONNX -- coord (%f, %f) (%f, %f)  image %ux%u\n", coord[0], coord[1], coord[2], coord[3], width, height);

		detections[numDetections].Instance   = numDetections;
		detections[numDetections].ClassID    = 0;
		detections[numDetections].Confidence = 1;
		detections[numDetections].Left       = coord[0];
		detections[numDetections].Top        = coord[1];
		detections[numDetections].Right      = coord[2];
		detections[numDetections].Bottom	  = coord[3];	
	
		numDetections++;
	}
	else
	{
		// cluster detections
		numDetections = clusterDetections(detections, width, height);
	}

	PROFILER_END(PROFILER_POSTPROCESS);

	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, rgba, width, height, detections, numDetections, overlay) )
			printf(LOG_TRT "detectNet::Detect() -- failed to render overlay\n");
	}
	
	return numDetections;
}



// clusterDetections
int detectNet::clusterDetections( Detection* detections, uint32_t width, uint32_t height )
{
	// cluster detection bboxes
	float* net_cvg   = mOutputs[OUTPUT_CVG].CPU;
	float* net_rects = mOutputs[OUTPUT_BBOX].CPU;
	
	const int ow  = DIMS_W(mOutputs[OUTPUT_BBOX].dims);	// number of columns in bbox grid in X dimension
	const int oh  = DIMS_H(mOutputs[OUTPUT_BBOX].dims);	// number of rows in bbox grid in Y dimension
	const int owh = ow * oh;							// total number of bbox in grid
	const int cls = GetNumClasses();					// number of object classes in coverage map
	
	const float cell_width  = /*width*/ DIMS_W(mInputDims) / ow;
	const float cell_height = /*height*/ DIMS_H(mInputDims) / oh;
	
	const float scale_x = float(width) / float(DIMS_W(mInputDims));
	const float scale_y = float(height) / float(DIMS_H(mInputDims));

#ifdef DEBUG_CLUSTERING	
	printf("input width %i height %i\n", (int)DIMS_W(mInputDims), (int)DIMS_H(mInputDims));
	printf("cells x %i  y %i\n", ow, oh);
	printf("cell width %f  height %f\n", cell_width, cell_height);
	printf("scale x %f  y %f\n", scale_x, scale_y);
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
					printf("rect x=%u y=%u  cvg=%f  %f %f   %f %f \n", x, y, coverage, x1, x2, y1, y2);
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


// from detectNet.cu
cudaError_t cudaDetectionOverlay( float4* input, float4* output, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections, float4* colors );

// Overlay
bool detectNet::Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections, uint32_t flags )
{
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	if( flags == 0 )
	{
		printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_NONE, returning false\n");
		return false;
	}

	// bounding box overlay
	if( flags & OVERLAY_BOX )
	{
		if( CUDA_FAILED(cudaDetectionOverlay((float4*)input, (float4*)output, width, height, detections, numDetections, (float4*)mClassColors[1])) )
			return false;
	}

	// class label overlay
	if( flags & OVERLAY_LABEL )
	{
		static cudaFont* font = NULL;

		// make sure the font object is created
		if( !font )
		{
			font = cudaFont::Create();
	
			if( !font )
			{
				printf(LOG_TRT "detectNet -- Overlay() was called with OVERLAY_FONT, but failed to create cudaFont()\n");
				return false;
			}
		}

		// draw each object's description
		std::vector< std::pair< std::string, int2 > > labels;

		for( uint32_t n=0; n < numDetections; n++ )
		{
			labels.push_back( std::pair<std::string, int2>( GetClassDesc(detections[n].ClassID),
												   make_int2(detections[n].Left, detections[n].Top) ) );
		}

		font->OverlayText((float4*)input, width, height, labels, make_float4(255,255,255,255));
	}
	
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// OverlayFlagsFromStr
uint32_t detectNet::OverlayFlagsFromStr( const char* str_user )
{
	if( !str_user )
		return OVERLAY_BOX;

	// copy the input string into a temporary array,
	// because strok modifies the string
	const size_t str_length = strlen(str_user);

	if( str_length == 0 )
		return OVERLAY_BOX;

	char* str = (char*)malloc(str_length + 1);

	if( !str )
		return OVERLAY_BOX;

	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
	{
		free(str);
		return OVERLAY_BOX;
	}

	// look for the tokens:  "box", "label", and "none"
	uint32_t flags = OVERLAY_NONE;

	while( token != NULL )
	{
		printf("%s\n", token);

		if( strcasecmp(token, "box") == 0 )
			flags |= OVERLAY_BOX;
		else if( strcasecmp(token, "label") == 0 || strcasecmp(token, "labels") == 0 )
			flags |= OVERLAY_LABEL;

		token = strtok(NULL, delimiters);
	}	

	free(str);
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
