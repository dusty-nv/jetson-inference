/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "tensorConvert.h"

#include "commandLine.h"
#include "cudaMappedMemory.h"

#include "mat33.h"


#define DEPTH_HISTOGRAM_CUDA


// constructor
depthNet::depthNet() : tensorNet()
{
	mNetworkType    = CUSTOM;
	mDepthRange     = NULL;
	mDepthEqualized = NULL;
	
	mHistogram      = NULL;
	mHistogramPDF   = NULL;
	mHistogramCDF   = NULL;
	mHistogramEDU   = NULL;
}


// destructor
depthNet::~depthNet()
{
	CUDA_FREE_HOST(mDepthEqualized);
	
#ifdef DEPTH_HISTOGRAM_CUDA
	CUDA_FREE_HOST(mDepthRange);
	
	CUDA_FREE(mHistogram);
	CUDA_FREE(mHistogramPDF);
	CUDA_FREE(mHistogramCDF);
	CUDA_FREE(mHistogramEDU);
#endif
}


// VisualizationFlagsFromStr
uint32_t depthNet::VisualizationFlagsFromStr( const char* str_user, uint32_t default_value )
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

		if( strcasecmp(token, "input") == 0 )
			flags |= VISUALIZE_INPUT;
		else if( strcasecmp(token, "depth") == 0 )
			flags |= VISUALIZE_DEPTH;

		token = strtok(NULL, delimiters);
	}	

	free(str);
	return flags;
}


// NetworkTypeFromStr
depthNet::NetworkType depthNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return depthNet::CUSTOM;

	depthNet::NetworkType type = depthNet::FCN_MOBILENET;

	if( strcasecmp(modelName, "mobilenet") == 0 || strcasecmp(modelName, "fcn-mobilenet") == 0 || strcasecmp(modelName, "fcn_mobilenet") == 0 || strcasecmp(modelName, "monodepth-fcn-mobilenet") == 0 || strcasecmp(modelName, "monodepth_fcn_mobilenet") == 0 )
		type = depthNet::FCN_MOBILENET;
	else if( strcasecmp(modelName, "resnet18") == 0 || strcasecmp(modelName, "fcn-resnet18") == 0 || strcasecmp(modelName, "fcn_resnet18") == 0 || strcasecmp(modelName, "monodepth-fcn-resnet18") == 0 || strcasecmp(modelName, "monodepth_fcn_resnet18") == 0 )
		type = depthNet::FCN_RESNET18;
	else if( strcasecmp(modelName, "resnet50") == 0 || strcasecmp(modelName, "fcn-resnet50") == 0 || strcasecmp(modelName, "fcn_resnet50") == 0 || strcasecmp(modelName, "monodepth-fcn-resnet50") == 0 || strcasecmp(modelName, "monodepth_fcn_resnet50") == 0 )
		type = depthNet::FCN_RESNET50;
	else
		type = depthNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* depthNet::NetworkTypeToStr( depthNet::NetworkType type )
{
	switch(type)
	{
		case FCN_MOBILENET:	return "MonoDepth-FCN-Mobilenet";
		case FCN_RESNET18:	return "MonoDepth-FCN-ResNet18";
		case FCN_RESNET50:	return "MonoDepth-FCN-ResNet50";
		default:			return "Custom";
	}
}


// Create
depthNet* depthNet::Create( depthNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	depthNet* net = NULL;
	
	if( networkType == FCN_MOBILENET )
		net = Create("networks/MonoDepth-FCN-Mobilenet/monodepth_fcn_mobilenet.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == FCN_RESNET18 )
		net = Create("networks/MonoDepth-FCN-ResNet18/monodepth_fcn_resnet18.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == FCN_RESNET50 )
		net = Create("networks/MonoDepth-FCN-ResNet50/monodepth_fcn_resnet50.onnx", DEPTHNET_DEFAULT_INPUT, DEPTHNET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	
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

	// allocate post-processing buffers
	if( !net->allocHistogramBuffers() )
		return NULL;

	return net;
}


// Create (UFF)
depthNet* depthNet::Create( const char* model_path, const char* input, 
					   const Dims3& inputDims, const char* output,
					   uint32_t maxBatchSize, precisionType precision,
				   	   deviceType device, bool allowGPUFallback )
{
	depthNet* net = new depthNet();
	
	if( !net )
		return NULL;
	
	printf("\n");
	printf("depthNet -- loading mono depth network model from:\n");
	printf("         -- model:      %s\n", model_path);
	printf("         -- input_blob  '%s'\n", input);
	printf("         -- output_blob '%s'\n", output);
	printf("         -- batch_size  %u\n\n", maxBatchSize);
	
	// create list of output names	
	std::vector<std::string> output_blobs;
	output_blobs.push_back(output);

	// increase workspace size for UFF
	net->mWorkspaceSize = 96 << 20;

	// load network
	if( !net->LoadNetwork(NULL, model_path, NULL, input, inputDims, output_blobs, 
					  maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf("depthNet -- failed to initialize.\n");
		return NULL;
	}

	// reorder UFF outputs with HWC dims (when C=1)
	if( net->mOutputs[0].dims.d[2] == 1 )
	{
		net->mOutputs[0].dims.d[2] = net->mOutputs[0].dims.d[1];
		net->mOutputs[0].dims.d[1] = net->mOutputs[0].dims.d[0];
		net->mOutputs[0].dims.d[0] = 1;
	}

	// load colormaps
	CUDA(cudaColormapInit());

	// allocate post-processing buffers
	if( !net->allocHistogramBuffers() )
		return NULL;
	
	return net;
}


// Create
depthNet* depthNet::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// Create
depthNet* depthNet::Create( const commandLine& cmdLine )
{
	depthNet* net = NULL;

	// obtain the network name
	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "fcn-mobilenet");
	
	// parse the network type
	const depthNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == depthNet::CUSTOM )
	{
		const char* input  = cmdLine.GetString("input_blob");
		const char* output = cmdLine.GetString("output_blob");

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


// allocHistogramBuffers
bool depthNet::allocHistogramBuffers()
{
	if( !cudaAllocMapped((void**)&mDepthEqualized, GetDepthFieldWidth() * GetDepthFieldHeight() * sizeof(float)) )
		return false;
	
#ifdef DEPTH_HISTOGRAM_CUDA
	if( !cudaAllocMapped((void**)&mDepthRange, sizeof(int2)) )
		return false;

	if( CUDA_FAILED(cudaMalloc((void**)&mHistogram, DEPTH_HISTOGRAM_BINS * sizeof(uint32_t))) )
		return false;
	
	if( CUDA_FAILED(cudaMalloc((void**)&mHistogramPDF, DEPTH_HISTOGRAM_BINS * sizeof(float))) )
		return false;
	
	if( CUDA_FAILED(cudaMalloc((void**)&mHistogramCDF, DEPTH_HISTOGRAM_BINS * sizeof(float))) )
		return false;
	
	if( CUDA_FAILED(cudaMalloc((void**)&mHistogramEDU, DEPTH_HISTOGRAM_BINS * sizeof(uint32_t))) )
		return false;
#endif
	
	return true;
}


// Process
bool depthNet::Process( void* input, uint32_t input_width, uint32_t input_height, imageFormat input_format )
{
	if( !input || input_width == 0 || input_height == 0 )
	{
		printf(LOG_TRT "depthNet::Process( 0x%p, %u, %u ) -> invalid parameters\n", input, input_width, input_height);
		return false;
	}

	if( !imageFormatIsRGB(input_format) )
	{
		imageFormatErrorMsg(LOG_TRT, "depthNet::Process()", input_format);
		return false;
	}
	
	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( IsModelType(MODEL_ONNX) )
	{
		// remap from [0,255] -> [0,1], no mean pixel subtraction or std dev applied
		if( CUDA_FAILED(cudaTensorNormMeanRGB(input, input_format, input_width, input_height, 
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
	else if( IsModelType(MODEL_UFF) )
	{
		// remap to planar BGR, apply mean pixel subtraction
		if( CUDA_FAILED(cudaTensorMeanBGR(input, input_format, input_width, input_height,
								    mInputs[0].CUDA, GetInputWidth(), GetInputHeight(),
								    make_float3(123.0, 115.0, 101.0),
								    GetStream())) )
		{
			printf(LOG_TRT "depthNet::Process() -- cudaPreImageNetMeanBGR() failed\n");
			return false;
		}
	}
	else
	{
		printf(LOG_TRT "depthNet::Process() -- support for models other than ONNX and UFF not implemented.\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);
	
#ifdef DEPTH_HISTOGRAM_CUDA
	if( !ProcessNetwork(false) )
		return false;
#else
	if( !ProcessNetwork(true) )
		return false;
#endif

	PROFILER_END(PROFILER_NETWORK);

	return true;
}


// Process
bool depthNet::Process( void* input, imageFormat input_format, void* output, imageFormat output_format, uint32_t width, uint32_t height, cudaColormapType colormap, cudaFilterMode filter )
{
	return Process(input, width, height, input_format, output, width, height, output_format, colormap, filter);
}


// Process
bool depthNet::Process( void* input, uint32_t input_width, uint32_t input_height, imageFormat input_format,
				    void* output, uint32_t output_width, uint32_t output_height, imageFormat output_format,
                        cudaColormapType colormap, cudaFilterMode filter )
{
	if( !Process(input, input_width, input_height, input_format) )
		return false;

	if( !Visualize(output, output_width, output_height, output_format, colormap, filter) )
		return false;

	return true;
}


// Visualize
bool depthNet::Visualize( void* output, uint32_t output_width, uint32_t output_height, imageFormat output_format,
				 	 cudaColormapType colormap, cudaFilterMode filter )
{
	if( !output || output_width == 0 || output_height == 0 )
	{
		printf(LOG_TRT "depthNet::Visualize( 0x%p, %u, %u ) -> invalid parameters\n", output, output_width, output_height);
		return false;
	}
	
	if( !imageFormatIsRGB(output_format) )
	{
		imageFormatErrorMsg(LOG_TRT, "depthNet::Visualize()", output_format);
		return false;
	}

	PROFILER_BEGIN(PROFILER_POSTPROCESS);

#ifdef DEPTH_HISTOGRAM_CUDA
	if( !histogramEqualizationCUDA() )
		return false;
#else
	CUDA(cudaStreamSynchronize(GetStream()));

	if( !histogramEqualization() )
		return false;
#endif

	PROFILER_END(PROFILER_POSTPROCESS);
	PROFILER_BEGIN(PROFILER_VISUALIZE);

	// apply color mapping to depth image
	if( CUDA_FAILED(cudaColormap(mDepthEqualized, GetDepthFieldWidth(), GetDepthFieldHeight(),
						    output, output_width, output_height, 
						    make_float2(0,255), FORMAT_DEFAULT, output_format,
						    colormap, filter, GetStream())) )
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

		if( !Visualize((void*)depthField, width, height, IMAGE_GRAY32F, COLORMAP_NONE, FILTER_LINEAR) )
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


