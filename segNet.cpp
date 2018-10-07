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
#include <cmath>


// constructor
segNet::segNet() : tensorNet()
{
	mClassColors[0] = NULL;	// cpu ptr
	mClassColors[1] = NULL;  // gpu ptr

	mClassMap[0] = NULL;
	mClassMap[1] = NULL;
}


// destructor
segNet::~segNet()
{
	
}


// Create
segNet* segNet::Create( NetworkType networkType, uint32_t maxBatchSize )
{
	if( networkType == FCN_ALEXNET_PASCAL_VOC )
		return Create("networks/FCN-Alexnet-Pascal-VOC/deploy.prototxt", "networks/FCN-Alexnet-Pascal-VOC/snapshot_iter_146400.caffemodel", "networks/FCN-Alexnet-Pascal-VOC/pascal-voc-classes.txt", "networks/FCN-Alexnet-Pascal-VOC/pascal-voc-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );
	else if( networkType == FCN_ALEXNET_SYNTHIA_CVPR16 )
		return Create("networks/FCN-Alexnet-SYNTHIA-CVPR16/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-CVPR16/snapshot_iter_1206700.caffemodel", "networks/FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-labels.txt", "networks/FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );
	else if( networkType == FCN_ALEXNET_SYNTHIA_SUMMER_HD )
		return Create("networks/FCN-Alexnet-SYNTHIA-Summer-HD/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/snapshot_iter_902888.caffemodel", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/synthia-seq-labels.txt", "networks/FCN-Alexnet-SYNTHIA-Summer-HD/synthia-seq-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );	
	else if( networkType == FCN_ALEXNET_SYNTHIA_SUMMER_SD )
		return Create("networks/FCN-Alexnet-SYNTHIA-Summer-SD/deploy.prototxt", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/snapshot_iter_431816.caffemodel", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/synthia-seq-labels.txt", "networks/FCN-Alexnet-SYNTHIA-Summer-SD/synthia-seq-train-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );		
	else if( networkType == FCN_ALEXNET_CITYSCAPES_HD )
		return Create("networks/FCN-Alexnet-Cityscapes-HD/deploy.prototxt", "networks/FCN-Alexnet-Cityscapes-HD/snapshot_iter_367568.caffemodel", "networks/FCN-Alexnet-Cityscapes-HD/cityscapes-labels.txt", "networks/FCN-Alexnet-Cityscapes-HD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );	
	else if( networkType == FCN_ALEXNET_CITYSCAPES_SD )
		return Create("networks/FCN-Alexnet-Cityscapes-SD/deploy.prototxt", "networks/FCN-Alexnet-Cityscapes-SD/snapshot_iter_2756640.caffemodel", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-labels.txt", "networks/FCN-Alexnet-Cityscapes-SD/cityscapes-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );
	//else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p_4ch )
	//	return Create("FCN-Alexnet-Aerial-FPV-4ch-720p/deploy.prototxt", "FCN-Alexnet-Aerial-FPV-4ch-720p/snapshot_iter_1777146.caffemodel", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-labels.txt", "FCN-Alexnet-Aerial-FPV-4ch-720p/fpv-deploy-colors.txt", "data", "score_fr_4classes", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );			
	else if( networkType == FCN_ALEXNET_AERIAL_FPV_720p )
		return Create("networks/FCN-Alexnet-Aerial-FPV-720p/fcn_alexnet.deploy.prototxt", "networks/FCN-Alexnet-Aerial-FPV-720p/snapshot_iter_10280.caffemodel", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-labels.txt", "networks/FCN-Alexnet-Aerial-FPV-720p/fpv-deploy-colors.txt", SEGNET_DEFAULT_INPUT, SEGNET_DEFAULT_OUTPUT, maxBatchSize );		
	else
		return NULL;
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
segNet* segNet::Create( const char* prototxt, const char* model, const char* labels_path, const char* colors_path, const char* input_blob, const char* output_blob, uint32_t maxBatchSize )
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
	
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs, maxBatchSize) )
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
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", filename);
		return false;
	}
	
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
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", filename);
		return false;
	}
	
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
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight );




// Overlay
bool segNet::Overlay( float* rgba, float* output, uint32_t width, uint32_t height, const char* ignore_class )
{
	if( !rgba || width == 0 || height == 0 || !output )
	{
		printf("segNet::Overlay( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight)) )
	{
		printf("segNet::Overlay() -- cudaPreImageNet failed\n");
		return false;
	}

	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "segNet::Overlay() -- failed to execute tensorRT context\n");
		return false;
	}

	PROFILER_REPORT();	// report total time, when profiling enabled

	
	// retrieve scores
	float* scores = mOutputs[0].CPU;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);
	const int s_c = DIMS_C(mOutputs[0].dims);
		
	//const float s_x = float(width) / float(s_w);		// TODO bug: this should use mWidth/mHeight dimensions, in case user dimensions are different
	//const float s_y = float(height) / float(s_h);
	const float s_x = float(s_w) / float(width);
	const float s_y = float(s_h) / float(height);

	// if desired, find the ID of the class to ignore (typically void)
	const int ignoreID = FindClassID(ignore_class);
	
	printf(LOG_GIE "segNet::Overlay -- s_w %i  s_h %i  s_c %i  s_x %f  s_y %f\n", s_w, s_h, s_c, s_x, s_y);
	printf(LOG_GIE "segNet::Overlay -- ignoring class '%s' id=%i\n", ignore_class, ignoreID);


	// find the argmax-classified class of each tile
	uint8_t* classMap = mClassMap[0];

	for( uint32_t y=0; y < s_h; y++ )
	{
		for( uint32_t x=0; x < s_w; x++ )
		{
			float p_max[3] = {-100000.0f, -100000.0f, -100000.0f };
			int   c_max[3] = { -1, -1, -1 };

			for( uint32_t c=0; c < s_c; c++ )	// classes
			{
				const float p = scores[c * s_w * s_h + y * s_w + x];

				if( c_max[0] < 0 || p > p_max[0] )
				{
					p_max[0] = p;
					c_max[0] = c;
				}
				else if( c_max[1] < 0 || p > p_max[1] )
				{
					p_max[1] = p;
					c_max[1] = c;
				}
				else if( c_max[2] < 0 || p > p_max[2] )
				{
					p_max[2] = p;
					c_max[2] = c;
				}
			}

			/*printf("%02u %u  class %i  %f  %s  class %i  %f  %s  class %i  %f  %s\n", x, y, 
				   c_max[0], p_max[0], (c_max[0] >= 0 && c_max[0] < GetNumClasses()) ? GetClassLabel(c_max[0]) : " ", 
				   c_max[1], p_max[1], (c_max[1] >= 0 && c_max[1] < GetNumClasses()) ? GetClassLabel(c_max[1]) : " ",
				   c_max[2], p_max[2], (c_max[2] >= 0 && c_max[2] < GetNumClasses()) ? GetClassLabel(c_max[2]) : " ");
			*/

			const int argmax = (c_max[0] == ignoreID) ? c_max[1] : c_max[0];

			classMap[y * s_w + x] = argmax;
		}
	}
	   
	// overlay pixels onto original
	for( uint32_t y=0; y < height; y++ )
	{
		for( uint32_t x=0; x < width; x++ )
		{
			const float cx = float(x) * s_x;
			const float cy = float(y) * s_y;

			const int x1 = floor(cx);
			const int y1 = floor(cy);
			
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

			

			const float x1d = cx - float(x1);
			const float y1d = cy - float(y1);
		
			const float x2d = 1.0f - x1d;
			const float y2d = 1.0f - y1d;

			const float x1f = 1.0f - x1d;
			const float y1f = 1.0f - y1d;

			const float x2f = 1.0f - x1f;
			const float y2f = 1.0f - y1f;

			int c_index = 0;

			/*if( y2d > y1d )
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

			float* px_in  = rgba +   (((y * width * 4) + x * 4));
			float* px_out = output + (((y * width * 4) + x * 4));

			const float alph = c_color[3] / 255.0f;
			const float inva = 1.0f - alph;

			px_out[0] = alph * c_color[0] + inva * px_in[0];
			px_out[1] = alph * c_color[1] + inva * px_in[1];
			px_out[2] = alph * c_color[2] + inva * px_in[2];
			px_out[3] = 255.0f;
		}
	}

	return true;
}

// Overlay and save the video
bool segNet::Overlay_save( float* rgba, float* output,float* output2, uint32_t width, uint32_t height, const char* ignore_class )
{
	if( !rgba || width == 0 || height == 0 || !output )
	{
		printf("segNet::Overlay( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight)) )
	{
		printf("segNet::Overlay() -- cudaPreImageNet failed\n");
		return false;
	}


	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };

	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "segNet::Overlay() -- failed to execute tensorRT context\n");
		return false;
	}

	PROFILER_REPORT();	// report total time, when profiling enabled


	// retrieve scores
	float* scores = mOutputs[0].CPU;

	const int s_w = DIMS_W(mOutputs[0].dims);
	const int s_h = DIMS_H(mOutputs[0].dims);
	const int s_c = DIMS_C(mOutputs[0].dims);

	//const float s_x = float(width) / float(s_w);		// TODO bug: this should use mWidth/mHeight dimensions, in case user dimensions are different
	//const float s_y = float(height) / float(s_h);
	//const float s_x = float(s_w) / float(mWidth);
	//const float s_y = float(s_h) / float(mHeight);
	const float s_x = float(s_w) / float(width);
	const float s_y = float(s_h) / float(height);

	// if desired, find the ID of the class to ignore (typically void)
	const int ignoreID = FindClassID(ignore_class);

	printf(LOG_GIE "segNet::Overlay -- width %i height %i\n",width,height);
	printf(LOG_GIE "segNet::Overlay -- mWidth %i mHeight %i\n",mWidth,mHeight);
	printf(LOG_GIE "segNet::Overlay -- s_w %i  s_h %i  s_c %i  s_x %f  s_y %f\n", s_w, s_h, s_c, s_x, s_y);
	printf(LOG_GIE "segNet::Overlay -- ignoring class '%s' id=%i\n", ignore_class, ignoreID);


	// find the argmax-classified class of each tile
	uint8_t* classMap = mClassMap[0];

	for( uint32_t y=0; y < s_h; y++ )
	{
		for( uint32_t x=0; x < s_w; x++ )
		{
			float p_max[3] = {-100000.0f, -100000.0f, -100000.0f };
			int   c_max[3] = { -1, -1, -1 };

			for( uint32_t c=0; c < s_c; c++ )	// classes
			{
				const float p = scores[c * s_w * s_h + y * s_w + x];

				if( c_max[0] < 0 || p > p_max[0] )
				{
					p_max[0] = p;
					c_max[0] = c;
				}
				else if( c_max[1] < 0 || p > p_max[1] )
				{
					p_max[1] = p;
					c_max[1] = c;
				}
				else if( c_max[2] < 0 || p > p_max[2] )
				{
					p_max[2] = p;
					c_max[2] = c;
				}
			}

			/*printf("%02u %u  class %i  %f  %s  class %i  %f  %s  class %i  %f  %s\n", x, y,
				   c_max[0], p_max[0], (c_max[0] >= 0 && c_max[0] < GetNumClasses()) ? GetClassLabel(c_max[0]) : " ",
				   c_max[1], p_max[1], (c_max[1] >= 0 && c_max[1] < GetNumClasses()) ? GetClassLabel(c_max[1]) : " ",
				   c_max[2], p_max[2], (c_max[2] >= 0 && c_max[2] < GetNumClasses()) ? GetClassLabel(c_max[2]) : " ");
			*/

			const int argmax = (c_max[0] == ignoreID) ? c_max[1] : c_max[0];

			classMap[y * s_w + x] = argmax;
		}
	}

	// overlay pixels onto original
	for( uint32_t y=0; y < height; y++ )
	{
		for( uint32_t x=0; x < width; x++ )
		{
			const float cx = float(x) * s_x;
			const float cy = float(y) * s_y;

			const int x1 = floor(cx);
			const int y1 = floor(cy);

			const int x2 = x1+1;
			const int y2 = y1+1;

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



			const float x1d = cx - float(x1);
			const float y1d = cy - float(y1);

			const float x2d = 1.0f - x1d;
			const float y2d = 1.0f - y1d;

			const float x1f = 1.0f - x1d;
			const float y1f = 1.0f - y1d;

			const float x2f = 1.0f - x1f;
			const float y2f = 1.0f - y1f;

			int c_index = 0;

			/*if( y2d > y1d )
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

			float* px_in  = rgba +   (((y * width * 4) + x * 4));
			float* px_out = output + (((y * width * 4) + x * 4));
			float* px_out2 = output2 + (((y * width * 4) + x * 4));

			const float alph = c_color[3] / 255.0f;
			const float inva = 1.0f - alph;

			px_out[0] = alph * c_color[0] + inva * px_in[0];
			px_out[1] = alph * c_color[1] + inva * px_in[1];
			px_out[2] = alph * c_color[2] + inva * px_in[2];
			px_out[3] = 255.0f;

			px_out2[0] = c_color[0];
			px_out2[1] = c_color[1];
			px_out2[2] = c_color[2];
			px_out2[3] = 255.0f;
		}
	}

	return true;
}


	
	
