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
 
#include "depthNet.h"
#include "imageNet.cuh"

#include "commandLine.h"
#include "cudaMappedMemory.h"

#include "mat33.h"


// constructor
depthNet::depthNet() : tensorNet()
{
	mNetworkType = CUSTOM;
}


// destructor
depthNet::~depthNet()
{

}


// NetworkTypeFromStr
depthNet::NetworkType depthNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return depthNet::CUSTOM;

	depthNet::NetworkType type = depthNet::MOBILENET;

	// ONNX models
	if( strcasecmp(modelName, "mobilenet") == 0 )
		type = depthNet::MOBILENET;
	else if( strcasecmp(modelName, "resnet-18") == 0 || strcasecmp(modelName, "resnet_18") == 0 || strcasecmp(modelName, "resnet18") == 0 )
		type = depthNet::RESNET_18;
	else if( strcasecmp(modelName, "resnet-50") == 0 || strcasecmp(modelName, "resnet_50") == 0 || strcasecmp(modelName, "resnet50") == 0 )
		type = depthNet::RESNET_50;
	else
		type = depthNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* depthNet::NetworkTypeToStr( depthNet::NetworkType type )
{
	switch(type)
	{
		case MOBILENET:	return "MonoDepth-Mobilenet";
		case RESNET_18:	return "MonoDepth-ResNet18";
		case RESNET_50:	return "MonoDepth-ResNet50";
		default:			return "Custom";
	}
}


// Create
depthNet* depthNet::Create( depthNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	depthNet* net = NULL;
	
	if( networkType == MOBILENET )
		net = Create("networks/DepthNet-Mobilenet/depth_mobilenet.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == RESNET_18 )
		net = Create("networks/DepthNet-ResNet18/depth_resnet18.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == RESNET_50 )
		net = Create("networks/DepthNet-ResNet50/depth_resnet50.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);

	if( net != NULL )
		net->mNetworkType = networkType;

	return net;
}


// Create
depthNet* depthNet::Create( const char* model_path, const char* input_blob, const char* output_blob,
					   uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback )
{
	depthNet* net = new depthNet();
	
	if( !net )
		return NULL;
	
	printf("\n");
	printf("depthNet -- loading mono depth network model from:\n");
	printf("         -- model:      %s\n", model_path);
	printf("         -- input_blob  '%s'\n", input_blob);
	printf("         -- output_blob '%s'\n", output_blob);
	printf("         -- batch_size  %u\n\n", maxBatchSize);
	
	// load network
	if( !net->LoadNetwork(NULL, model_path, NULL, input_blob, output_blob, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("depthNet -- failed to initialize.\n");
		return NULL;
	}

	// load colormaps
	CUDA(cudaColormapInit());

	// return network
	return net;
}


// Create
depthNet* depthNet::Create( int argc, char** argv )
{
	depthNet* net = NULL;

	// obtain the network name
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "mobilenet");
	
	// enable verbose mode if desired
	if( cmdLine.GetFlag("verbose") )
		tensorNet::EnableVerbose();

	// parse the network type
	const depthNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == depthNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input ) 	input  = DEPTHNET_DEFAULT_INPUT;
		if( !output )  output = DEPTHNET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = DEFAULT_MAX_BATCH_SIZE;

		net = depthNet::Create(modelName, input, output, maxBatchSize);
	}
	else
	{
		// create from pretrained model
		net = depthNet::Create(type);
	}

	if( !net )
		return NULL;

	// enable layer profiling if desired
	if( cmdLine.GetFlag("profile") )
		net->EnableLayerProfiler();

	return net;
}


// Process
bool depthNet::Process( float* input, uint32_t input_width, uint32_t input_height )
{
	if( !input || input_width == 0 || input_height == 0 )
	{
		printf(LOG_TRT "depthNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", input, input_width, input_height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_ONNX) )
	{
		// remap from [0,255] -> [0,1], no mean pixel subtraction or std dev applied
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)input, input_width, input_height, 
										   mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.0f, 0.0f, 0.0f),
										   make_float3(1.0f, 1.0f, 1.0f), 
										   GetStream())) )
		{
			printf(LOG_TRT "depthNet::Process() -- cudaPreImageNetNormMeanRGB() failed\n");
			return false;
		}
	}
	else
	{
		printf(LOG_TRT "depthNet::Process() -- support for models other than ONNX not implemented.\n");
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
bool depthNet::Process( float* input, float* output, uint32_t width, uint32_t height, cudaColormapType colormap, cudaFilterMode filter )
{
	return Process(input, width, height, output, width, height, colormap, filter);
}


// Process
bool depthNet::Process( float* input, uint32_t input_width, uint32_t input_height,
				    float* output, uint32_t output_width, uint32_t output_height, 
                        cudaColormapType colormap, cudaFilterMode filter )
{
	if( !Process(input, input_width, input_height) )
		return false;

	if( !Visualize(output, output_width, output_height, colormap, filter) )
		return false;

	return true;
}


// Visualize
bool depthNet::Visualize( float* output, uint32_t output_width, uint32_t output_height,
				 	 cudaColormapType colormap, cudaFilterMode filter )
{
	if( !output || output_width == 0 || output_height == 0 )
	{
		printf(LOG_TRT "depthNet::Visualize( 0x%p, %u, %u ) -> invalid parameters\n", output, output_width, output_height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// apply color mapping to depth image
	if( CUDA_FAILED(cudaColormap(GetDepthField(), GetDepthFieldWidth(), GetDepthFieldHeight(),
						    output, output_width, output_height, mDepthRange, 
						    colormap, filter, FORMAT_DEFAULT, GetStream())) )
	{
		printf(LOG_TRT "depthNet::Visualize() -- failed to map depth image with cudaColormap()\n");
		return false; 
	}

	PROFILER_END(PROFILER_VISUALIZE);
	return true;
}


// SavePointCloud
bool depthNet::SavePointCloud( const char* filename, float* imgRGBA, uint32_t width, uint32_t height,
						 const float2& focalLength, const float2& principalPoint )
{
	if( !filename || width == 0 || height == 0 )
	{
		printf(LOG_TRT "depthNet::SavePointCloud() -- invalid parameters\n");
		return false;
	}

	const bool has_rgb = (imgRGBA != NULL);
	const uint32_t numPoints = width * height;

	// create the PCD file
	FILE* file = fopen(filename, "w");

	if( !file )
	{
		printf(LOG_TRT "depthNet::SavePointCloud() -- failed to create %s\n", filename);
		return false;
	}

	// write the PCD header
	fprintf(file, "# .PCD v0.7 - Point Cloud Data file format\n");
	fprintf(file, "VERSION 0.7\n");

	if( has_rgb )
	{
		fprintf(file, "FIELDS x y z rgb\n");
		fprintf(file, "SIZE 4 4 4 4\n");
		fprintf(file, "TYPE F F F U\n");
	}
	else
	{
		fprintf(file, "FIELDS x y z\n");
		fprintf(file, "SIZE 4 4 4\n");
		fprintf(file, "TYPE F F F\n");
	}

	fprintf(file, "COUNT 1 1 1 1\n");
	fprintf(file, "WIDTH %u\n", numPoints);
	fprintf(file, "HEIGHT 1\n");
	fprintf(file, "VIEWPOINT 0 0 0 1 0 0 0\n");
	fprintf(file, "POINTS %u\n", numPoints);
	fprintf(file, "DATA ascii\n");

	// if RGB mode, upsample the depth field to match
	float* depthField = NULL;

	if( has_rgb )
	{
		if( !cudaAllocMapped((void**)&depthField, numPoints * sizeof(float)) )
		{
			printf(LOG_TRT "depthNet::SavePointCloud() -- failed to allocate CUDA memory for depth field (%u points)\n", numPoints);
			return false;
		}

		if( !Visualize(depthField, width, height, COLORMAP_NONE, FILTER_LINEAR) )
		{
			printf(LOG_TRT "depthNet::SavePointCloud() -- failed to upsample depth field\n");
			return false;
		}

		CUDA(cudaDeviceSynchronize());
	}

	// extract the point cloud
	for( int y=0; y < height; y++ )
	{
		for( int x=0; x < width; x++ )
		{
			const float depth = depthField[y * width + x];

			const float p_x = (float(x) - principalPoint.x) * depth / focalLength.x;
			const float p_y = (float(y) - principalPoint.y) * depth / focalLength.y * -1.0f;
			const float p_z = depth * -1.0f;	// invert y/z for model viewing

			fprintf(file, "%f %f %f", p_x, p_y, p_z);

			if( has_rgb )
			{
				const float4 rgba = ((float4*)imgRGBA)[y * width + x];
				const uint32_t rgb = (uint32_t(rgba.x) << 16 |
		      					  uint32_t(rgba.y) << 8 | 
								  uint32_t(rgba.z));
			
				fprintf(file, " %u", rgb);
			}

			fprintf(file, "\n");
		}
	}

	// free resources
	if( has_rgb && depthField != NULL )
		CUDA(cudaFreeHost(depthField));
	
	fclose(file);
	return true;
}


// SavePointCloud
bool depthNet::SavePointCloud( const char* filename )
{
	return SavePointCloud(filename, NULL, GetDepthFieldWidth(), GetDepthFieldHeight());
}


// SavePointCloud
bool depthNet::SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height )
{
	const float f_w = (float)width;
	const float f_h = (float)height;

	return SavePointCloud(filename, rgba, width, height, make_float2(f_h, f_h),
					  make_float2(f_w * 0.5f, f_h * 0.5f));
}


// SavePointCloud
bool depthNet::SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height,
					 	 const float intrinsicCalibration[3][3] )
{
	return SavePointCloud(filename, rgba, width, height,
					  make_float2(intrinsicCalibration[0][0], intrinsicCalibration[1][1]),
					  make_float2(intrinsicCalibration[0][2], intrinsicCalibration[1][2]));
}

// SavePointCloud
bool depthNet::SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height,
					 	 const char* intrinsicCalibrationPath )
{
	if( !intrinsicCalibrationPath )
		return SavePointCloud(filename, rgba, width, height);

	// open the camera calibration file
	FILE* file = fopen(intrinsicCalibrationPath, "r");

	if( !file )
	{
		printf(LOG_TRT "depthNet::SavePointCloud() -- failed to open calibration file %s\n", intrinsicCalibrationPath);
		return false;
	}
 
	// parse the 3x3 calibration matrix
	float K[3][3];

	for( int n=0; n < 3; n++ )
	{
		char str[512];

		if( !fgets(str, 512, file) )
		{
			printf(LOG_TRT "depthNet::SavePointCloud() -- failed to read line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}

		const int len = strlen(str);

		if( len <= 0 )
		{
			printf(LOG_TRT "depthNet::SavePointCloud() -- invalid line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}

		if( str[len-1] == '\n' )
			str[len-1] = 0;

		if( sscanf(str, "%f %f %f", &K[n][0], &K[n][1], &K[n][2]) != 3 )
		{
			printf(LOG_TRT "depthNet::SavePointCloud() -- failed to parse line %i from calibration file %s\n", n+1, intrinsicCalibrationPath);
			return false;
		}
	}

	// close the file
	fclose(file);

	// dump the matrix
	printf(LOG_TRT "depthNet::SavePointCloud() -- loaded intrinsic camera calibration from %s\n", intrinsicCalibrationPath);
	mat33_print(K, "K");

	// proceed with processing the point cloud
	return SavePointCloud(filename, rgba, width, height, K);
}


