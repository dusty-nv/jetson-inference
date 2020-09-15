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
 
#include "segNet.h"
#include "tensorConvert.h"

#include "cudaMappedMemory.h"
#include "cudaOverlay.h"
#include "cudaResize.h"
#include "cudaFont.h"

#include "commandLine.h"
#include "filesystem.h"
#include "imageIO.h"


// constructor
segNet::segNet() : tensorNet()
{
	mLastInputImg    = NULL;
	mLastInputWidth  = 0;
	mLastInputHeight = 0;
	mLastInputFormat = IMAGE_UNKNOWN;

	mColorsAlphaSet = NULL;
	mClassColors    = NULL;
	mClassMap       = NULL;

	mNetworkType = SEGNET_CUSTOM;
}


// destructor
segNet::~segNet()
{
	CUDA_FREE_HOST(mClassColors);
	CUDA_FREE_HOST(mClassMap);
	
	if( mColorsAlphaSet != NULL )
	{
		free(mColorsAlphaSet);
		mColorsAlphaSet = NULL;
	}
}


// VisualizationFlagsFromStr
uint32_t segNet::VisualizationFlagsFromStr( const char* str_user, uint32_t default_value )
{
	if( !str_user )
		return default_value;

	// copy the input string into a temporary array,
	// because strok modifies the string
	const size_t str_length = strlen(str_user);

	if( str_length == 0 )
		return default_value;

	char* str = (char*)malloc(str_length + 1);

	if( !str )
		return default_value;

	strcpy(str, str_user);

	// tokenize string by delimiters ',' and '|'
	const char* delimiters = ",|";
	char* token = strtok(str, delimiters);

	if( !token )
	{
		free(str);
		return default_value;
	}

	// look for the tokens:  "overlay", "mask"
	uint32_t flags = 0;

	while( token != NULL )
	{
		//printf("%s\n", token);

		if( strcasecmp(token, "overlay") == 0 )
			flags |= VISUALIZE_OVERLAY;
		else if( strcasecmp(token, "mask") == 0 )
			flags |= VISUALIZE_MASK;

		token = strtok(NULL, delimiters);
	}	

	free(str);
	return flags;
}


// FilterModeFromStr
segNet::FilterMode segNet::FilterModeFromStr( const char* str, FilterMode default_value )
{
	if( !str )
		return default_value;

	if( strcasecmp(str, "point") == 0 )
		return segNet::FILTER_POINT;
	else if( strcasecmp(str, "linear") == 0 )
		return segNet::FILTER_LINEAR;

	return default_value;
}


// NetworkTypeFromStr
segNet::NetworkType segNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return segNet::SEGNET_CUSTOM;

	segNet::NetworkType type = segNet::FCN_RESNET18_VOC_320x320;

	// ONNX models
	if( strcasecmp(modelName, "fcn-resnet18-cityscapes-512x256") == 0 || strcasecmp(modelName, "fcn-resnet18-cityscapes") == 0 )
		type = segNet::FCN_RESNET18_CITYSCAPES_512x256;
	else if( strcasecmp(modelName, "fcn-resnet18-cityscapes-1024x512") == 0 )
		type = segNet::FCN_RESNET18_CITYSCAPES_1024x512;
	else if( strcasecmp(modelName, "fcn-resnet18-cityscapes-2048x1024") == 0 )
		type = segNet::FCN_RESNET18_CITYSCAPES_2048x1024;
	else if( strcasecmp(modelName, "fcn-resnet18-deepscene-576x320") == 0 || strcasecmp(modelName, "fcn-resnet18-deepscene") == 0)
		type = segNet::FCN_RESNET18_DEEPSCENE_576x320;
	else if( strcasecmp(modelName, "fcn-resnet18-deepscene-864x480") == 0 )
		type = segNet::FCN_RESNET18_DEEPSCENE_864x480;
	else if( strcasecmp(modelName, "fcn-resnet18-mhp-512x320") == 0 || strcasecmp(modelName, "fcn-resnet18-mhp") == 0 )
		type = segNet::FCN_RESNET18_MHP_512x320;
	else if( strcasecmp(modelName, "fcn-resnet18-mhp-640x360") == 0 )
		type = segNet::FCN_RESNET18_MHP_640x360;
	else if( strcasecmp(modelName, "fcn-resnet18-voc-320x320") == 0 || strcasecmp(modelName, "fcn-resnet18-pascal-voc-320x320") == 0 || strcasecmp(modelName, "fcn-resnet18-voc") == 0 || strcasecmp(modelName, "fcn-resnet18-pascal-voc") == 0 )
		type = segNet::FCN_RESNET18_VOC_320x320;
	else if( strcasecmp(modelName, "fcn-resnet18-voc-512x320") == 0 || strcasecmp(modelName, "fcn-resnet18-pascal-voc-512x320") == 0 )
		type = segNet::FCN_RESNET18_VOC_512x320;
	else if( strcasecmp(modelName, "fcn-resnet18-sun-512x400") == 0 || strcasecmp(modelName, "fcn-resnet18-sun-rgbd-512x400") == 0 || strcasecmp(modelName, "fcn-resnet18-sun") == 0 || strcasecmp(modelName, "fcn-resnet18-sunrgb") == 0 )
		type = segNet::FCN_RESNET18_SUNRGB_512x400;
	else if( strcasecmp(modelName, "fcn-resnet18-sun-640x512") == 0 || strcasecmp(modelName, "fcn-resnet18-sun-rgbd-640x512") == 0 )
		type = segNet::FCN_RESNET18_SUNRGB_640x512;

	// legacy models
	else if( strcasecmp(modelName, "fcn-alexnet-cityscapes-sd") == 0 || strcasecmp(modelName, "fcn-alexnet-cityscapes") == 0 )
		type = segNet::FCN_ALEXNET_CITYSCAPES_SD;
	else if( strcasecmp(modelName, "fcn-alexnet-cityscapes-hd") == 0 )
		type = segNet::FCN_ALEXNET_CITYSCAPES_HD;
	else if( strcasecmp(modelName, "fcn-alexnet-pascal-voc") == 0 )
		type = segNet::FCN_ALEXNET_PASCAL_VOC;
	else if( strcasecmp(modelName, "synthia-cvpr16") == 0 || strcasecmp(modelName, "fcn-alexnet-synthia-cvpr16") == 0 )
		type = segNet::FCN_ALEXNET_SYNTHIA_CVPR16;
	else if( strcasecmp(modelName, "synthia-summer-sd") == 0 || strcasecmp(modelName, "fcn-alexnet-synthia-summer-sd") == 0 )
		type = segNet::FCN_ALEXNET_SYNTHIA_SUMMER_SD;
	else if( strcasecmp(modelName, "synthia-summer-hd") == 0 || strcasecmp(modelName, "fcn-alexnet-synthia-summer-hd") == 0 )
		type = segNet::FCN_ALEXNET_SYNTHIA_SUMMER_HD;
	else if( strcasecmp(modelName, "aerial-fpv") == 0 || strcasecmp(modelName, "aerial-fpv-720p") == 0 || strcasecmp(modelName, "fcn-alexnet-aerial-fpv-720p") == 0 )
		type = segNet::FCN_ALEXNET_AERIAL_FPV_720p;
	else
		type = segNet::SEGNET_CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* segNet::NetworkTypeToStr( segNet::NetworkType type )
{
	switch(type)
	{
		// ONNX models
		case FCN_RESNET18_CITYSCAPES_512x256:	return "fcn-resnet18-cityscapes-512x256";
		case FCN_RESNET18_CITYSCAPES_1024x512:	return "fcn-resnet18-cityscapes-1024x512";
		case FCN_RESNET18_CITYSCAPES_2048x1024:	return "fcn-resnet18-cityscapes-2048x1024";
		case FCN_RESNET18_DEEPSCENE_576x320:	return "fcn-resnet18-deepscene-576x320";
		case FCN_RESNET18_DEEPSCENE_864x480:	return "fcn-resnet18-deepscene-864x480";
		case FCN_RESNET18_MHP_512x320:		return "fcn-resnet18-mhp-512x320";
		case FCN_RESNET18_MHP_640x360:		return "fcn-resnet18-mhp-640x360";
		case FCN_RESNET18_VOC_320x320:		return "fcn-resnet18-voc-320x320";
		case FCN_RESNET18_VOC_512x320:		return "fcn-resnet18-voc-512x320";
		case FCN_RESNET18_SUNRGB_512x400:		return "fcn-resnet18-sun-512x400";
		case FCN_RESNET18_SUNRGB_640x512:		return "fcn-resnet18-sun-640x512";

		// legacy models
		case FCN_ALEXNET_PASCAL_VOC:			return "fcn-alexnet-pascal-voc";
		case FCN_ALEXNET_SYNTHIA_CVPR16:		return "fcn-alexnet-synthia-cvpr16";
		case FCN_ALEXNET_SYNTHIA_SUMMER_HD:	return "fcn-alexnet-synthia-summer-hd";
		case FCN_ALEXNET_SYNTHIA_SUMMER_SD:	return "fcn-alexnet-synthia-summer-sd";
		case FCN_ALEXNET_CITYSCAPES_HD:		return "fcn-alexnet-cityscapes-hd";
		case FCN_ALEXNET_CITYSCAPES_SD:		return "fcn-alexnet-cityscapes-sd";
		case FCN_ALEXNET_AERIAL_FPV_720p:		return "fcn-alexnet-aerial-fpv-720p";
		default:							return "custom segNet";
	}
}


// Create
segNet* segNet::Create( NetworkType networkType, uint32_t maxBatchSize,
				    precisionType precision, deviceType device, bool allowGPUFallback )
{
	segNet* net = NULL;

	#define LOAD_ONNX(x) Create(NULL, "networks/" x "/fcn_resnet18.onnx", "networks/" x "/classes.txt", "networks/" x "/colors.txt", "input_0", "output_0", maxBatchSize, precision, device, allowGPUFallback )

	// ONNX models
	if( networkType == FCN_RESNET18_CITYSCAPES_512x256 )
		net = LOAD_ONNX("FCN-ResNet18-Cityscapes-512x256");
	else if( networkType == FCN_RESNET18_CITYSCAPES_1024x512 )
		net = LOAD_ONNX("FCN-ResNet18-Cityscapes-1024x512");
	else if( networkType == FCN_RESNET18_CITYSCAPES_2048x1024 )
		net = LOAD_ONNX("FCN-ResNet18-Cityscapes-2048x1024");
	else if( networkType == FCN_RESNET18_DEEPSCENE_576x320 )
		net = LOAD_ONNX("FCN-ResNet18-DeepScene-576x320");
	else if( networkType == FCN_RESNET18_DEEPSCENE_864x480 )
		net = LOAD_ONNX("FCN-ResNet18-DeepScene-864x480");
	else if( networkType == FCN_RESNET18_MHP_512x320 )
		net = LOAD_ONNX("FCN-ResNet18-MHP-512x320");
	else if( networkType == FCN_RESNET18_MHP_640x360 )
		net = LOAD_ONNX("FCN-ResNet18-MHP-640x360");
	else if( networkType == FCN_RESNET18_VOC_320x320 )
		net = LOAD_ONNX("FCN-ResNet18-Pascal-VOC-320x320");
	else if( networkType == FCN_RESNET18_VOC_512x320 )
		net = LOAD_ONNX("FCN-ResNet18-Pascal-VOC-512x320");
	else if( networkType == FCN_RESNET18_SUNRGB_512x400 )
		net = LOAD_ONNX("FCN-ResNet18-SUN-RGBD-512x400");
	else if( networkType == FCN_RESNET18_SUNRGB_640x512 )
		net = LOAD_ONNX("FCN-ResNet18-SUN-RGBD-640x512");

	// legacy models
	else if( networkType == FCN_ALEXNET_PASCAL_VOC )
		net = Create("networks/FCN-Alexnet-Pascal-VOC/deploy.prototxt", "networks/FCN-Alexnet-Pascal-VOC/snapshot_iter_146400.caffemodel", "networks/FCN-Alexnet-Pascal-VOC/pascal-voc-classes.txt", "networks/FCN-Alexnet-Pascal-VOC/pascal-voc-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == FCN_ALEXNET_SYNTHIA_CVPR16 )
		net = Create("networks/FCN-Alexnet-SYNTHIA-CVPR16/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-CVPR16/snapshot_iter_1206700.caffemodel", "networks/FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-labels.txt", "networks/FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == FCN_ALEXNET_SYNTHIA_SUMMER_HD )
		net = Create("networks/FCN-Alexnet-SYNTHIA-Summer-HD/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/snapshot_iter_902888.caffemodel", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/synthia-seq-labels.txt", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/synthia-seq-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == FCN_ALEXNET_SYNTHIA_SUMMER_SD )
		net = Create("networks/FCN-Alexnet-SYNTHIA-Summer-SD/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/snapshot_iter_431816.caffemodel", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/synthia-seq-labels.txt", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/synthia-seq-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == FCN_ALEXNET_CITYSCAPES_HD )
		net = Create("networks/FCN-Alexnet-Cityscapes-HD/deploy.prototxt", "networks/FCN-Alexnet-Cityscapes-HD/snapshot_iter_367568.caffemodel", "networks/FCN-Alexnet-Cityscapes-HD/cityscapes-labels.txt", "networks/FCN-Alexnet-Cityscapes-HD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == FCN_ALEXNET_CITYSCAPES_SD )
		net = Create("networks/FCN-Alexnet-Cityscapes-SD/deploy.prototxt", "networks/FCN-Alexnet-Cityscapes-SD/snapshot_iter_2756640.caffemodel", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-labels.txt", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	//else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p_4ch )
	//	net = Create("FCN-Alexnet-Aerial-FPV-4ch-720p/deploy.prototxt", "FCN-Alexnet-Aerial-FPV-4ch-720p/snapshot_iter_1777146.caffemodel", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-labels.txt", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-deploy-colors.txt", "data", "score_fr_4classes", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );			
	else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p )
		net = Create("networks/FCN-Alexnet-Aerial-FPV-720p/fcn_alexnet.deploy.prototxt", "networks/FCN-Alexnet-Aerial-FPV-720p/snapshot_iter_10280.caffemodel", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-labels.txt", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else
		return NULL;

	if( net != NULL )
		net->mNetworkType = networkType;

	return net;
}


// Create
segNet* segNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
segNet* segNet::Create( const commandLine& cmdLine )
{
	segNet* net = NULL;

	// obtain the model name
	const char* modelName = cmdLine.GetString("model");

	if( !modelName )
		modelName = cmdLine.GetString("network", "fcn-resnet18-voc-320x320");

	// parse the model type
	const segNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == SEGNET_CUSTOM )
	{
		const char* prototxt = cmdLine.GetString("prototxt");
		const char* labels   = cmdLine.GetString("labels");
		const char* colors   = cmdLine.GetString("colors");
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input = SEGNET_DEFAULT_INPUT;
		if( !output )  output = SEGNET_DEFAULT_OUTPUT;

		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;
		
		net = segNet::Create(prototxt, modelName, labels, colors, input, output, maxBatchSize);
	}
	else
	{
		// create segnet from pretrained model
		net = segNet::Create(type);
	}

	if( !net )
		return NULL;

	// save the legend if desired
	const char* legend = cmdLine.GetString("legend");

	if( legend != NULL )
		net->saveClassLegend(legend);

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	// set overlay alpha value
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", SEGNET_DEFAULT_ALPHA));

	return net;
}


// Create
segNet* segNet::Create( const char* prototxt, const char* model, const char* labels_path, const char* colors_path, 
				    const char* input_blob, const char* output_blob, uint32_t maxBatchSize,
				    precisionType precision, deviceType device, bool allowGPUFallback )
{
	// create segmentation model
	segNet* net = new segNet();
	
	if( !net )
		return NULL;

	LogInfo("\n");
	LogInfo("segNet -- loading segmentation network model from:\n");
	LogInfo("       -- prototxt:   %s\n", prototxt);
	LogInfo("       -- model:      %s\n", model);
	LogInfo("       -- labels:     %s\n", labels_path);
	LogInfo("       -- colors:     %s\n", colors_path);
	LogInfo("       -- input_blob  '%s'\n", input_blob);
	LogInfo("       -- output_blob '%s'\n", output_blob);
	LogInfo("       -- batch_size  %u\n\n", maxBatchSize);
	
	//net->EnableProfiler();	
	//net->EnableDebug();
	//net->DisableFP16();		// debug;

	// load network
	std::vector<std::string> output_blobs;
	output_blobs.push_back(output_blob);
	
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, maxBatchSize,
					  precision, device, allowGPUFallback) )
	{
		LogError(LOG_TRT "segNet -- failed to load.\n");
		return NULL;
	}
	
	// initialize array of class colors
	const uint32_t numClasses = net->GetNumClasses();
	
	if( !cudaAllocMapped((void**)&net->mClassColors, numClasses * sizeof(float4)) )
		return NULL;
	
	for( uint32_t n=0; n < numClasses; n++ )
	{
		net->mClassColors[n*4+0] = 255.0f;	// r
		net->mClassColors[n*4+1] = 0.0f;	// g
		net->mClassColors[n*4+2] = 0.0f;	// b
		net->mClassColors[n*4+3] = 255.0f;	// a
	}
	
	net->mColorsAlphaSet = (bool*)malloc(numClasses * sizeof(bool));

	if( !net->mColorsAlphaSet )
	{
		printf(LOG_TRT "segNet -- failed to allocate class colors alpha flag array\n");
		return NULL;
	}

	memset(net->mColorsAlphaSet, 0, numClasses * sizeof(bool));
	
	// initialize array of classified argmax
	const int s_w = DIMS_W(net->mOutputs[0].dims);
	const int s_h = DIMS_H(net->mOutputs[0].dims);
	const int s_c = DIMS_C(net->mOutputs[0].dims);
		
	LogVerbose(LOG_TRT "segNet outputs -- s_w %i  s_h %i  s_c %i\n", s_w, s_h, s_c);

	if( !cudaAllocMapped((void**)&net->mClassMap, s_w * s_h * sizeof(uint8_t)) )
		return NULL;

	// load class info
	net->loadClassLabels(labels_path);
	net->loadClassColors(colors_path);

	return net;
}


// loadClassColors
bool segNet::loadClassColors( const char* filename )
{
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "segNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "segNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class colors
	char str[512];
	int  idx = 0;

	while( fgets(str, 512, f) != NULL )
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
			LogVerbose(LOG_TRT "segNet -- class %02i  color %i %i %i %i\n", idx, r, g, b, a);
			SetClassColor(idx, r, g, b, a);
			idx++; 
		}
	}
	
	fclose(f);
	
	LogVerbose(LOG_TRT "segNet -- loaded %i class colors\n", idx);
	
	if( idx == 0 )
		return false;
	
	return true;
}


// loadClassLabels
bool segNet::loadClassLabels( const char* filename )
{
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "segNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "segNet -- failed to open %s\n", path.c_str());
		return false;
	}
	
	// read class labels
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int len = strlen(str);
		
		if( len > 0 )
		{
			if( str[len-1] == '\n' )
				str[len-1] = 0;

			LogVerbose(LOG_TRT "segNet -- class %02zu  label '%s'\n", mClassLabels.size(), str);
			mClassLabels.push_back(str);
		}
	}
	
	fclose(f);
	
	LogVerbose(LOG_TRT "segNet -- loaded %zu class labels\n", mClassLabels.size());
	
	if( mClassLabels.size() == 0 )
		return false;
	
	mClassPath = path;
	return true;
}


// saveClassLegend
bool segNet::saveClassLegend( const char* filename )
{
	const int2 colorSize = make_int2(50,25);
	const int2 xyPadding = make_int2(10,5);

	const float4 bgColor = make_float4(255,255,255,255);
	const float4 fgColor = make_float4(0,0,0,255);

	// validate arguments
	if( !filename )
		return false;

	const uint32_t numClasses = GetNumClasses();

	if( numClasses == 0 )
		return false;

	// load the font
	cudaFont* font = cudaFont::Create(16);

	if( !font )
		return false;

	// determine the max size of class labels
	int2 maxTextExtents = make_int2(0,0);

	for( uint32_t n=0; n < numClasses; n++ )
	{
		char str[256];
		sprintf(str, "%2d %s", n, GetClassDesc(n));

		const int4 textExtents = font->TextExtents(str);

		if( textExtents.z > maxTextExtents.x )
			maxTextExtents.x = textExtents.z;

		if( textExtents.w > maxTextExtents.y )
			maxTextExtents.y = textExtents.w;
	}

	//if( colorSize.y > maxTextExtents.y )
	//	maxTextExtents.y = colorSize.y;

	// allocate image to store the legend
	const int imgWidth = maxTextExtents.x + colorSize.x + xyPadding.x * 3;
	const int imgHeight = (colorSize.y + xyPadding.y) * numClasses + xyPadding.y * 2;
	
	float4* img = NULL;

	if( !cudaAllocMapped((void**)&img, imgWidth * imgHeight * 4 * sizeof(float)) )
		return false;

	// fill the legend's background color
	#define FILL_RECT(color, x1, y1, x2, y2) \
		for( int y=y1; y < y2; y++ )		 \
			for( int x=x1; x < x2; x++ )	 \
				img[y*imgWidth+x] = color;

	FILL_RECT(bgColor, 0, 0, imgWidth, imgHeight);

	// render each class entry
	int yPosition = xyPadding.y * 2;

	for( uint32_t n=0; n < numClasses; n++ )
	{
		char str[256];
		sprintf(str, "%2d %s", n, GetClassDesc(n));

		// render the class text
		font->OverlayText(img, imgWidth, imgHeight, str, xyPadding.x, yPosition);
		CUDA(cudaDeviceSynchronize());

		// fill the class color
		float* classColor  = GetClassColor(n);
		const float4 color = make_float4(classColor[0], classColor[1], classColor[2], 255);
		
		const int colorX = maxTextExtents.x + xyPadding.x * 2;
		const int colorY = yPosition - ((colorSize.y - maxTextExtents.y) / 2);

		FILL_RECT(color, colorX, colorY, colorX + colorSize.x, colorY + colorSize.y);

		// advance the position
		yPosition += colorSize.y + xyPadding.y;
	}

	// save the image
	const bool result = saveImageRGBA(filename, img, imgWidth, imgHeight);

	CUDA(cudaFreeHost(img));
	delete font;
	return result;
}


// SetClassColor
void segNet::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[i+0] = r;
	mClassColors[i+1] = g;
	mClassColors[i+2] = b;
	mClassColors[i+3] = a;

	mColorsAlphaSet[classIndex] = (a == 255) ? false : true;
}


// SetOverlayAlpha
void segNet::SetOverlayAlpha( float alpha, bool explicit_exempt )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
	{
		if( !explicit_exempt || !mColorsAlphaSet[n] /*mClassColors[n*4+3] == 255*/ )
			mClassColors[n*4+3] = alpha;
	}
}


// FindClassID
int segNet::FindClassID( const char* label_name )
{
	if( !label_name )
		return -1;

	const uint32_t numLabels = mClassLabels.size();

	for( uint32_t n=0; n < numLabels; n++ )
	{
		if( strcasecmp(label_name, mClassLabels[n].c_str()) == 0 )
			return n;
	}

	return -1;
}


// Process
bool segNet::Process( float* rgba, uint32_t width, uint32_t height, const char* ignore_class )
{
	return Process(rgba, width, height, IMAGE_RGBA32F, ignore_class);
}


// Process
bool segNet::Process( void* image, uint32_t width, uint32_t height, imageFormat format, const char* ignore_class )
{
	if( !image || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "segNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", image, width, height);
		return false;
	}

	if( !imageFormatIsRGB(format) )
	{
		LogError(LOG_TRT "segNet -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_TRT "          supported formats are:\n");
		LogError(LOG_TRT "              * rgb8\n");		
		LogError(LOG_TRT "              * rgba8\n");		
		LogError(LOG_TRT "              * rgb32f\n");		
		LogError(LOG_TRT "              * rgba32f\n");

		return cudaErrorInvalidValue;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_ONNX) )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization, mean pixel subtraction and standard deviation
		if( CUDA_FAILED(cudaTensorNormMeanRGB(image, format, width, height,
									   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
									   make_float2(0.0f, 1.0f), 
									   make_float3(0.485f, 0.456f, 0.406f),
									   make_float3(0.229f, 0.224f, 0.225f), 
									   GetStream())) )
		{
			LogError(LOG_TRT "segNet::Process() -- cudaTensorNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		// downsample and convert to band-sequential BGR
		if( CUDA_FAILED(cudaTensorMeanBGR(image, format, width, height, 
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float3(0,0,0), GetStream())) )
		{
			LogError(LOG_TRT "segNet::Process() -- cudaTensorMeanBGR() failed\n");
			return false;
		}
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
	// process with TensorRT
	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// generate argmax classification map
	if( !classify(ignore_class) )
		return false;

	PROFILER_END(PROFILER_POSTPROCESS);

	// cache pointer to last image processed
	mLastInputImg    = image;
	mLastInputWidth  = width;
	mLastInputHeight = height;
	mLastInputFormat = format;

	return true;
}


// argmax classification
bool segNet::classify( const char* ignore_class )
{
	// retrieve scores
	float* scores = mOutputs[0].CPU;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);
	const int s_c = DIMS_C(mOutputs[0].dims);
		
	//const float s_x = float(width) / float(s_w);		// TODO bug: this should use mWidth/mHeight dimensions, in case user dimensions are different
	//const float s_y = float(height) / float(s_h);
	const float s_x = float(s_w) / float(GetInputWidth());
	const float s_y = float(s_h) / float(GetInputHeight());


	// if desired, find the ID of the class to ignore (typically void)
	const int ignoreID = FindClassID(ignore_class);
	
	//printf(LOG_TRT "segNet::Process -- s_w %i  s_h %i  s_c %i  s_x %f  s_y %f\n", s_w, s_h, s_c, s_x, s_y);
	//printf(LOG_TRT "segNet::Process -- ignoring class '%s' id=%i\n", ignore_class, ignoreID);


	// find the argmax-classified class of each tile
	uint8_t* classMap = mClassMap;

	for( uint32_t y=0; y < s_h; y++ )
	{
		for( uint32_t x=0; x < s_w; x++ )
		{
			float p_max = -100000.0f;
			int   c_max = -1;

			for( int c=0; c < s_c; c++ )
			{
				// skip ignoreID
				if( c == ignoreID )
					continue;

				// check if this class score is higher
				const float p = scores[c * s_w * s_h + y * s_w + x];

				if( c_max < 0 || p > p_max )
				{
					p_max = p;
					c_max = c;
				}
			}

			classMap[y * s_w + x] = c_max;
			//printf("(%u, %u) -> class %i\n", x, y, (uint32_t)c_max);
		}
	}

	return true;
}


// Mask (binary)
bool segNet::Mask( uint8_t* output, uint32_t out_width, uint32_t out_height )
{
	if( !output || out_width == 0 || out_height == 0 )
	{
		LogError(LOG_TRT "segNet::Mask( 0x%p, %u, %u ) -> invalid parameters\n", output, out_width, out_height); 
		return false;
	}	

	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// retrieve classification map
	uint8_t* classMap = mClassMap;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);
		
	if( out_width == s_w && out_height == s_h )
	{
		memcpy(output, classMap, s_w * s_h * sizeof(uint8_t));
	}
	else
	{
		const float s_x = float(s_w) / float(out_width);
		const float s_y = float(s_h) / float(out_height);

		// overlay pixels onto original
		for( uint32_t y=0; y < out_height; y++ )
		{
			for( uint32_t x=0; x < out_width; x++ )
			{
				const int cx = float(x) * s_x;
				const int cy = float(y) * s_y;

				// get the class ID of this cell
				const uint8_t classIdx = classMap[cy * s_w + cx];

				// output the pixel
				output[y * out_width + x] = classIdx;
			}
		}
	}
	
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// Mask (colorized)
bool segNet::Mask( float* output, uint32_t width, uint32_t height, FilterMode filter )
{
	return Mask(output, width, height, IMAGE_RGBA32F, filter);
}


// Mask (colorized)
bool segNet::Mask( void* output, uint32_t width, uint32_t height, imageFormat format, FilterMode filter )
{
	if( !output || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "segNet::Mask( 0x%p, %u, %u ) -> invalid parameters\n", output, width, height); 
		return false;
	}	

	// filter in point or linear
	if( filter == FILTER_POINT )
		return overlayPoint(NULL, 0, 0, IMAGE_UNKNOWN, output, width, height, format, true);
	else if( filter == FILTER_LINEAR )
		return overlayLinear(NULL, 0, 0, IMAGE_UNKNOWN, output, width, height, format, true);

	return false;
}


// Overlay
bool segNet::Overlay( float* output, uint32_t width, uint32_t height, FilterMode filter )
{
	return Overlay(output, width, height, IMAGE_RGBA32F, filter);
}


// Overlay
bool segNet::Overlay( void* output, uint32_t width, uint32_t height, imageFormat format, segNet::FilterMode filter )
{
	if( !output || width == 0 || height == 0 )
	{
		LogError(LOG_TRT "segNet::Overlay( 0x%p, %u, %u ) -> invalid parameters\n", output, width, height); 
		return false;
	}	
	
	if( !mLastInputImg )
	{
		LogError(LOG_TRT "segNet -- Process() must be called before Overlay()\n");
		return false;
	}

	// filter in point or linear
	if( filter == FILTER_POINT )
		return overlayPoint(mLastInputImg, mLastInputWidth, mLastInputHeight, mLastInputFormat, output, width, height, format, false);
	else if( filter == FILTER_LINEAR )
		return overlayLinear(mLastInputImg, mLastInputWidth, mLastInputHeight, mLastInputFormat, output, width, height, format, false);

	return false;
}


#define OVERLAY_CUDA 

// declaration from segNet.cu
cudaError_t cudaSegOverlay( void* input, uint32_t in_width, uint32_t in_height,
				        void* output, uint32_t out_width, uint32_t out_height, imageFormat format,
					   float4* class_colors, uint8_t* scores, const int2& scores_dim,
					   bool filter_linear, bool mask_only, cudaStream_t stream );


// overlayLinear
bool segNet::overlayPoint( void* input, uint32_t in_width, uint32_t in_height, imageFormat in_format, void* output, uint32_t out_width, uint32_t out_height, imageFormat out_format, bool mask_only )
{
	if( input != NULL && in_format != out_format )
	{
		LogError(LOG_TRT "segNet -- input image format (%s) and overlay/mask image format (%s) don't match\n", imageFormatToStr(in_format), imageFormatToStr(out_format));
		return false;
	}

	PROFILER_BEGIN(PROFILER_VISUALIZE);

#ifdef OVERLAY_CUDA
	// generate overlay on the GPU
	if( CUDA_FAILED(cudaSegOverlay(input, in_width, in_height, output, out_width, out_height, out_format,
							 (float4*)mClassColors, mClassMap, make_int2(DIMS_W(mOutputs[0].dims), DIMS_H(mOutputs[0].dims)),
							 false, mask_only, GetStream())) )
	{
		LogError(LOG_TRT "segNet -- failed to process %ux%u overlay/mask with CUDA\n", out_width, out_height);
		return false;
	}
#else
	// retrieve classification map
	uint8_t* classMap = mClassMap;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);

	const float s_x = float(s_w) / float(out_width);
	const float s_y = float(s_h) / float(out_height);


	// overlay pixels onto original
	for( uint32_t y=0; y < out_height; y++ )
	{
		for( uint32_t x=0; x < out_width; x++ )
		{
			const int cx = float(x) * s_x;
			const int cy = float(y) * s_y;

			// get the class ID of this cell
			const uint8_t classIdx = classMap[cy * s_w + cx];

			// find the color of this class
			float* c_color = GetClassColor(classIdx);

			// output the pixel
			float* px_out = output + (((y * out_width * 4) + x * 4));

			if( mask_only )
			{
				// only draw the segmentation mask
				px_out[0] = c_color[0];
				px_out[1] = c_color[1];
				px_out[2] = c_color[2];
				px_out[3] = 255.0f;
			}
			else
			{
				// alpha blend with input image
				const uint32_t x_in = float(x) / float(out_width) * float(in_width);
				const uint32_t y_in = float(y) / float(out_height) * float(in_height);

				float* px_in = input + (((y_in * in_width * 4) + x_in * 4));
				
				const float alph = c_color[3] / 255.0f;
				const float inva = 1.0f - alph;

				px_out[0] = alph * c_color[0] + inva * px_in[0];
				px_out[1] = alph * c_color[1] + inva * px_in[1];
				px_out[2] = alph * c_color[2] + inva * px_in[2];
				px_out[3] = 255.0f;
			}
		}
	}
#endif
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// overlayLinear
bool segNet::overlayLinear( void* input, uint32_t in_width, uint32_t in_height, imageFormat in_format, void* output, uint32_t out_width, uint32_t out_height, imageFormat out_format, bool mask_only )
{
	if( input != NULL && in_format != out_format )
	{
		LogError(LOG_TRT "segNet -- input image format (%s) and overlay/mask image format (%s) don't match\n", imageFormatToStr(in_format), imageFormatToStr(out_format));
		return false;
	}

	PROFILER_BEGIN(PROFILER_VISUALIZE);

#ifdef OVERLAY_CUDA
	// generate overlay on the GPU
	if( CUDA_FAILED(cudaSegOverlay(input, in_width, in_height, output, out_width, out_height, out_format,
							 (float4*)mClassColors, mClassMap, make_int2(DIMS_W(mOutputs[0].dims), DIMS_H(mOutputs[0].dims)),
							 true, mask_only, GetStream())) )
	{
		LogError(LOG_TRT "segNet -- failed to process %ux%u overlay/mask with CUDA\n", out_width, out_height);
		return false;
	}
#else
	// retrieve classification map
	uint8_t* classMap = mClassMap;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);

	const float s_x = float(s_w) / float(out_width);
	const float s_y = float(s_h) / float(out_height);


	// overlay pixels onto original
	for( uint32_t y=0; y < out_height; y++ )
	{
		for( uint32_t x=0; x < out_width; x++ )
		{
			const float cx = float(x) * s_x;	
			const float cy = float(y) * s_y;

			const int x1 = int(cx);
			const int y1 = int(cy);
			
			const int x2 = x1 + 1;
			const int y2 = y1 + 1;

			#define CHK_BOUNDS(x, y)		( (y < 0 ? 0 : (y >= (s_h - 1) ? (s_h - 1) : y)) * s_w + (x < 0 ? 0 : (x >= (s_w - 1) ? (s_w - 1) : x)) )

			/*const uint8_t classIdx[] = { classMap[y1 * s_w + x1],
								    classMap[y1 * s_w + x2],
								    classMap[y2 * s_w + x2],
								    classMap[y2 * s_w + x1] };*/

			const uint8_t classIdx[] = { classMap[CHK_BOUNDS(x1, y1)],
								    classMap[CHK_BOUNDS(x2, y1)],
								    classMap[CHK_BOUNDS(x2, y2)],
								    classMap[CHK_BOUNDS(x1, y2)] };


			float* cc[] = { GetClassColor(classIdx[0]),
						 GetClassColor(classIdx[1]),
						 GetClassColor(classIdx[2]),
						 GetClassColor(classIdx[3]) };

			
			// compute bilinear weights
			const float x1d = cx - float(x1);
			const float y1d = cy - float(y1);
		
			const float x2d = 1.0f - x1d;
			const float y2d = 1.0f - y1d;

			const float x1f = 1.0f - x1d;
			const float y1f = 1.0f - y1d;

			const float x2f = 1.0f - x1f;
			const float y2f = 1.0f - y1f;

			/*int c_index = 0;

			if( y2d > y1d )
			{
				if( x2d > y2d )			c_index = 2;
				else 					c_index = 3;
			}
			else
			{
				if( x2d > y2d )			c_index = 1;
				else						c_index = 0;
			}*/
			
			//float* c_color = GetClassColor(classIdx[c_index]);
			//printf("x %u y %u cx %f cy %f  x1d %f y1d %f  x2d %f y2d %f  c %i\n", x, y, cx, cy, x1d, y1d, x2d, y2d, c_index);

			float c_color[] = { cc[0][0] * x1f * y1f + cc[1][0] * x2f * y1f + cc[2][0] * x2f * y2f + cc[3][0] * x1f * y2f,
						     cc[0][1] * x1f * y1f + cc[1][1] * x2f * y1f + cc[2][1] * x2f * y2f + cc[3][1] * x1f * y2f,
						     cc[0][2] * x1f * y1f + cc[1][2] * x2f * y1f + cc[2][2] * x2f * y2f + cc[3][2] * x1f * y2f,
						     cc[0][3] * x1f * y1f + cc[1][3] * x2f * y1f + cc[2][3] * x2f * y2f + cc[3][3] * x1f * y2f };

			// output the pixel
			float* px_out = output + (((y * out_width * 4) + x * 4));

			if( mask_only )
			{
				// only draw the segmentation mask
				px_out[0] = c_color[0];
				px_out[1] = c_color[1];
				px_out[2] = c_color[2];
				px_out[3] = 255.0f;
			}
			else
			{
				// alpha blend with input image
				const int x_in = float(x) / float(out_width) * float(in_width);
				const int y_in = float(y) / float(out_height) * float(in_height);

				float* px_in = input + (((y_in * in_width * 4) + x_in * 4));
				
				const float alph = c_color[3] / 255.0f;
				const float inva = 1.0f - alph;

				px_out[0] = alph * c_color[0] + inva * px_in[0];
				px_out[1] = alph * c_color[1] + inva * px_in[1];
				px_out[2] = alph * c_color[2] + inva * px_in[2];
				px_out[3] = 255.0f;
			}
		}
	}
#endif
	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}
	
	
