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

#include "cudaMappedMemory.h"
#include "cudaOverlay.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"


// constructor
segNet::segNet() : tensorNet()
{
	mLastInputImg    = NULL;
	mLastInputWidth  = 0;
	mLastInputHeight = 0;

	mClassColors[0] = NULL;	// cpu ptr
	mClassColors[1] = NULL;  // gpu ptr

	mClassMap[0] = NULL;
	mClassMap[1] = NULL;

	mNetworkType = SEGNET_CUSTOM;
}


// destructor
segNet::~segNet()
{
	
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

	segNet::NetworkType type = segNet::FCN_ALEXNET_CITYSCAPES_HD;

	if( strcasecmp(modelName, "cityscapes-sd") == 0 || strcasecmp(modelName, "fcn-alexnet-cityscapes-sd") == 0 )
		type = segNet::FCN_ALEXNET_CITYSCAPES_SD;
	else if( strcasecmp(modelName, "cityscapes") == 0 || strcasecmp(modelName, "cityscapes-hd") == 0 || strcasecmp(modelName, "fcn-alexnet-cityscapes-hd") == 0 )
		type = segNet::FCN_ALEXNET_CITYSCAPES_HD;
	else if( strcasecmp(modelName, "pascal-voc") == 0 || strcasecmp(modelName, "fcn-alexnet-pascal-voc") == 0 )
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


// Create
segNet* segNet::Create( NetworkType networkType, uint32_t maxBatchSize,
				    precisionType precision, deviceType device, bool allowGPUFallback )
{
	segNet* net = NULL;

	if( networkType == FCN_ALEXNET_PASCAL_VOC )
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
		net = Create("networks/FCN-Alexnet-Cityscapes-SD/deploy.prototxt", "networks/FCN-Alexnet-Cityscapes-SD/snapshot_iter_114860.caffemodel", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-labels.txt", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	//else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p_4ch )
	//	net = Create("FCN-Alexnet-Aerial-FPV-4ch-720p/deploy.prototxt", "FCN-Alexnet-Aerial-FPV-4ch-720p/snapshot_iter_1777146.caffemodel", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-labels.txt", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-deploy-colors.txt", "data", "score_fr_4classes", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );			
	else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p )
		net = Create("networks/FCN-Alexnet-Aerial-FPV-720p/fcn_alexnet.deploy.prototxt", "networks/FCN-Alexnet-Aerial-FPV-720p/snapshot_iter_10280.caffemodel", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-labels.txt", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else
		return NULL;

	if( net != NULL )
		net->mNetworkType = networkType;
}


// Create
segNet* segNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("model");

	if( !modelName )
	{
		modelName = "fcn-alexnet-cityscapes-hd";

		if( argc > 3 )
			modelName = argv[3];	

		segNet::NetworkType type = segNet::SEGNET_CUSTOM;

		if( strcasecmp(modelName, "fcn-alexnet-cityscapes-sd") == 0 || strcasecmp(modelName, "fcn-alexnet-cityscapes") == 0 )
			type = segNet::FCN_ALEXNET_CITYSCAPES_SD;
		else if( strcasecmp(modelName, "fcn-alexnet-cityscapes-hd") == 0 )
			type = segNet::FCN_ALEXNET_CITYSCAPES_HD;
		else if( strcasecmp(modelName, "fcn-alexnet-pascal-voc") == 0 )
			type = segNet::FCN_ALEXNET_PASCAL_VOC;
		else if( strcasecmp(modelName, "fcn-alexnet-synthia-cvpr16") == 0 )
			type = segNet::FCN_ALEXNET_SYNTHIA_CVPR16;
		else if( strcasecmp(modelName, "fcn-alexnet-synthia-summer-sd") == 0 || strcasecmp(modelName, "fcn-alexnet-synthia-summer") == 0)
			type = segNet::FCN_ALEXNET_SYNTHIA_SUMMER_SD;
		else if( strcasecmp(modelName, "fcn-alexnet-synthia-summer-hd") == 0 )
			type = segNet::FCN_ALEXNET_SYNTHIA_SUMMER_HD;
		else if( strcasecmp(modelName, "fcn-alexnet-aerial-fpv-720p") == 0 )
			type = segNet::FCN_ALEXNET_AERIAL_FPV_720p;
		/*else if( strcasecmp(modelName, "fcn-alexnet-aerial-fpv-720p-4ch") == 0 )
			type = segNet::FCN_ALEXNET_AERIAL_FPV_720p_4ch;
		else if( strcasecmp(modelName, "fcn-alexnet-aerial-fpv-720p-21ch") == 0 )
			type = segNet::FCN_ALEXNET_AERIAL_FPV_720p_21ch;*/

		// create segnet from pretrained model
		return segNet::Create(type);
	}
	else
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
			maxBatchSize = 2;
		
		return segNet::Create(prototxt, modelName, labels, colors, input, output, maxBatchSize);
	}
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

	printf("\n");
	printf("segNet -- loading segmentation network model from:\n");
	printf("       -- prototxt:   %s\n", prototxt);
	printf("       -- model:      %s\n", model);
	printf("       -- labels:     %s\n", labels_path);
	printf("       -- colors:     %s\n", colors_path);
	printf("       -- input_blob  '%s'\n", input_blob);
	printf("       -- output_blob '%s'\n", output_blob);
	printf("       -- batch_size  %u\n\n", maxBatchSize);
	
	//net->EnableProfiler();	
	//net->EnableDebug();
	//net->DisableFP16();		// debug;

	// load network
	std::vector<std::string> output_blobs;
	output_blobs.push_back(output_blob);
	
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, maxBatchSize,
					  precision, device, allowGPUFallback) )
	{
		printf("segNet -- failed to initialize.\n");
		return NULL;
	}
	
	// initialize array of class colors
	const uint32_t numClasses = net->GetNumClasses();
	
	if( !cudaAllocMapped((void**)&net->mClassColors[0], (void**)&net->mClassColors[1], numClasses * sizeof(float4)) )
		return NULL;
	
	for( uint32_t n=0; n < numClasses; n++ )
	{
		net->mClassColors[0][n*4+0] = 255.0f;	// r
		net->mClassColors[0][n*4+1] = 0.0f;	// g
		net->mClassColors[0][n*4+2] = 0.0f;	// b
		net->mClassColors[0][n*4+3] = 255.0f;	// a
	}
	
	// initialize array of classified argmax
	const int s_w = DIMS_W(net->mOutputs[0].dims);
	const int s_h = DIMS_H(net->mOutputs[0].dims);
	const int s_c = DIMS_C(net->mOutputs[0].dims);
		
	printf(LOG_GIE "segNet outputs -- s_w %i  s_h %i  s_c %i\n", s_w, s_h, s_c);

	if( !cudaAllocMapped((void**)&net->mClassMap[0], (void**)&net->mClassMap[1], s_w * s_h * sizeof(uint8_t)) )
		return NULL;

	// load class info
	net->loadClassColors(colors_path);
	net->loadClassLabels(labels_path);
	
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
		printf("segNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", path.c_str());
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
			printf("segNet -- class %02i  color %i %i %i %i\n", idx, r, g, b, a);
			SetClassColor(idx, r, g, b, a);
			idx++; 
		}
	}
	
	fclose(f);
	
	printf("segNet -- loaded %i class colors\n", idx);
	
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
		printf("segNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", path.c_str());
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

			printf("segNet -- class %02zu  label '%s'\n", mClassLabels.size(), str);
			mClassLabels.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("segNet -- loaded %zu class labels\n", mClassLabels.size());
	
	if( mClassLabels.size() == 0 )
		return false;
	
	mClassPath = path;
	return true;
}


// SetClassColor
void segNet::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[0][i+0] = r;
	mClassColors[0][i+1] = g;
	mClassColors[0][i+2] = b;
	mClassColors[0][i+3] = a;
}


// SetGlobalAlpha
void segNet::SetGlobalAlpha( float alpha, bool explicit_exempt )
{
	const uint32_t numClasses = GetNumClasses();

	for( uint32_t n=0; n < numClasses; n++ )
	{
		if( !explicit_exempt || mClassColors[0][n*4+3] == 255 )
			mClassColors[0][n*4+3] = alpha;
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



// declaration from imageNet.cu
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, cudaStream_t stream );	



// Process
bool segNet::Process( float* rgba, uint32_t width, uint32_t height, const char* ignore_class )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("segNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
	{
		printf("segNet::Process() -- cudaPreImageNet failed\n");
		return false;
	}

	
	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "segNet::Process() -- failed to execute TensorRT context\n");
		return false;
	}

	PROFILER_REPORT();	// report total time, when profiling enabled


	// generate argmax classification map
	if( !classify(ignore_class) )
		return false;


	// cache pointer to last image processed
	mLastInputImg = rgba;
	mLastInputWidth = width;
	mLastInputHeight = height;

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
	const float s_x = float(s_w) / float(mWidth);
	const float s_y = float(s_h) / float(mHeight);


	// if desired, find the ID of the class to ignore (typically void)
	const int ignoreID = FindClassID(ignore_class);
	
	//printf(LOG_GIE "segNet::Process -- s_w %i  s_h %i  s_c %i  s_x %f  s_y %f\n", s_w, s_h, s_c, s_x, s_y);
	//printf(LOG_GIE "segNet::Process -- ignoring class '%s' id=%i\n", ignore_class, ignoreID);


	// find the argmax-classified class of each tile
	uint8_t* classMap = mClassMap[0];

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
		}
	}

	return true;
}


// Mask (binary)
bool segNet::Mask( uint8_t* output, uint32_t out_width, uint32_t out_height )
{
	if( !output || out_width == 0 || out_height == 0 )
	{
		printf("segNet::Mask( 0x%p, %u, %u ) -> invalid parameters\n", output, out_width, out_height); 
		return false;
	}	

	// retrieve classification map
	uint8_t* classMap = mClassMap[0];

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

			// output the pixel
			output[y * out_width + x] = classIdx;
		}
	}

	return true;
}


// Mask (colorized)
bool segNet::Mask( float* output, uint32_t width, uint32_t height, FilterMode filter )
{
	if( !output || width == 0 || height == 0 )
	{
		printf("segNet::Mask( 0x%p, %u, %u ) -> invalid parameters\n", output, width, height); 
		return false;
	}	

	// filter in point or linear
	if( filter == FILTER_POINT )
		return overlayPoint(NULL, 0, 0, output, width, height, true);
	else if( filter == FILTER_LINEAR )
		return overlayLinear(NULL, 0, 0, output, width, height, true);

	return false;
}


// Overlay
bool segNet::Overlay( float* output, uint32_t width, uint32_t height, FilterMode filter )
{
	if( !output || width == 0 || height == 0 )
	{
		printf("segNet::Overlay( 0x%p, %u, %u ) -> invalid parameters\n", output, width, height); 
		return false;
	}	
	
	if( !mLastInputImg )
	{
		printf(LOG_TRT "segNet -- Process() must be called before Overlay()\n");
		return false;
	}

	// filter in point or linear
	if( filter == FILTER_POINT )
		return overlayPoint(mLastInputImg, mLastInputWidth, mLastInputHeight, output, width, height, false);
	else if( filter == FILTER_LINEAR )
		return overlayLinear(mLastInputImg, mLastInputWidth, mLastInputHeight, output, width, height, false);

	return false;
}



// overlayLinear
bool segNet::overlayPoint( float* input, uint32_t in_width, uint32_t in_height, float* output, uint32_t out_width, uint32_t out_height, bool mask )
{
	// retrieve classification map
	uint8_t* classMap = mClassMap[0];

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

			if( mask )
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

	return true;
}


// overlayLinear
bool segNet::overlayLinear( float* input, uint32_t in_width, uint32_t in_height, float* output, uint32_t out_width, uint32_t out_height, bool mask )
{
	// retrieve classification map
	uint8_t* classMap = mClassMap[0];

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

			if( mask )
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

	return true;
}
	
	
