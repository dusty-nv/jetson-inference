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
#include "imageNet.cuh"

#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include "commandLine.h"
#include "filesystem.h"


// constructor
imageNet::imageNet() : tensorNet()
{
	mOutputClasses = 0;
	mNetworkType   = CUSTOM;
}


// destructor
imageNet::~imageNet()
{

}


// Create
imageNet* imageNet::Create( imageNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(networkType, maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf(LOG_TRT "imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	net->mNetworkType = networkType;
	return net;
}


// Create
imageNet* imageNet::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
					   const char* class_path, const char* input, const char* output, uint32_t maxBatchSize,
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(prototxt_path, model_path, mean_binary, class_path, input, output, maxBatchSize, precision, device, allowGPUFallback) )
	{
		printf(LOG_TRT "imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// init
bool imageNet::init( imageNet::NetworkType networkType, uint32_t maxBatchSize, 
				 precisionType precision, deviceType device, bool allowGPUFallback )
{
	if( networkType == imageNet::ALEXNET )
		return init( "networks/alexnet.prototxt", "networks/bvlc_alexnet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::GOOGLENET )
		return init( "networks/googlenet.prototxt", "networks/bvlc_googlenet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::GOOGLENET_12 )
		return init( "networks/GoogleNet-ILSVRC12-subset/deploy.prototxt", "networks/GoogleNet-ILSVRC12-subset/snapshot_iter_184080.caffemodel", NULL, "networks/GoogleNet-ILSVRC12-subset/labels.txt", IMAGENET_DEFAULT_INPUT, "softmax", maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::RESNET_18 )
		return init( "networks/ResNet-18/deploy.prototxt", "networks/ResNet-18/ResNet-18.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == imageNet::RESNET_50 )
		return init( "networks/ResNet-50/deploy.prototxt", "networks/ResNet-50/ResNet-50.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else if( networkType == imageNet::RESNET_101 )
		return init( "networks/ResNet-101/deploy.prototxt", "networks/ResNet-101/ResNet-101.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::RESNET_152 )
		return init( "networks/ResNet-152/deploy.prototxt", "networks/ResNet-152/ResNet-152.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::VGG_16 )
		return init( "networks/VGG-16/deploy.prototxt", "networks/VGG-16/VGG-16.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );
	else if( networkType == imageNet::VGG_19 )
		return init( "networks/VGG-19/deploy.prototxt", "networks/VGG-19/VGG-19.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );		
	else if( networkType == imageNet::INCEPTION_V4 )
		return init( "networks/Inception-v4/deploy.prototxt", "networks/Inception-v4/Inception-v4.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback );	
	else
		return NULL;
}


// init
bool imageNet::init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, 
				const char* input, const char* output, uint32_t maxBatchSize,
				precisionType precision, deviceType device, bool allowGPUFallback )
{
	if( /*!prototxt_path ||*/ !model_path || !class_path || !input || !output )
		return false;

	printf("\n");
	printf("imageNet -- loading classification network model from:\n");
	printf("         -- prototxt     %s\n", prototxt_path);
	printf("         -- model        %s\n", model_path);
	printf("         -- class_labels %s\n", class_path);
	printf("         -- input_blob   '%s'\n", input);
	printf("         -- output_blob  '%s'\n", output);
	printf("         -- batch_size   %u\n\n", maxBatchSize);

	/*
	 * load and parse googlenet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( prototxt_path, model_path, mean_binary, input, output, 
						    maxBatchSize, precision, device, allowGPUFallback ) )
	{
		printf(LOG_TRT "failed to load %s\n", model_path);
		return false;
	}

	printf(LOG_TRT "%s loaded\n", model_path);

	/*
	 * load synset classnames
	 */
	mOutputClasses = DIMS_C(mOutputs[0].dims);
	
	if( !loadClassInfo(class_path) || mClassSynset.size() != mOutputClasses || mClassDesc.size() != mOutputClasses )
	{
		printf("imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mOutputClasses);
		return false;
	}
	
	printf("%s initialized.\n", model_path);
	return true;
}
			

// NetworkTypeFromStr
imageNet::NetworkType imageNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return imageNet::CUSTOM;

	imageNet::NetworkType type = imageNet::GOOGLENET;

	if( strcasecmp(modelName, "alexnet") == 0 )
		type = imageNet::ALEXNET;
	else if( strcasecmp(modelName, "googlenet") == 0 )
		type = imageNet::GOOGLENET;
	else if( strcasecmp(modelName, "googlenet-12") == 0 || strcasecmp(modelName, "googlenet_12") == 0 )
		type = imageNet::GOOGLENET_12;
	else if( strcasecmp(modelName, "resnet-18") == 0 || strcasecmp(modelName, "resnet_18") == 0 || strcasecmp(modelName, "resnet18") == 0 )
		type = imageNet::RESNET_18;
	else if( strcasecmp(modelName, "resnet-50") == 0 || strcasecmp(modelName, "resnet_50") == 0 || strcasecmp(modelName, "resnet50") == 0 )
		type = imageNet::RESNET_50;
	else if( strcasecmp(modelName, "resnet-101") == 0 || strcasecmp(modelName, "resnet_101") == 0 || strcasecmp(modelName, "resnet101") == 0 )
		type = imageNet::RESNET_101;
	else if( strcasecmp(modelName, "resnet-152") == 0 || strcasecmp(modelName, "resnet_152") == 0 || strcasecmp(modelName, "resnet152") == 0 )
		type = imageNet::RESNET_152;
	else if( strcasecmp(modelName, "vgg-16") == 0 || strcasecmp(modelName, "vgg_16") == 0 || strcasecmp(modelName, "vgg16") == 0 )
		type = imageNet::VGG_16;
	else if( strcasecmp(modelName, "vgg-19") == 0 || strcasecmp(modelName, "vgg_19") == 0 || strcasecmp(modelName, "vgg19") == 0 )
		type = imageNet::VGG_19;
	else if( strcasecmp(modelName, "inception-v4") == 0 || strcasecmp(modelName, "inception_v4") == 0 || strcasecmp(modelName, "inceptionv4") == 0 )
		type = imageNet::INCEPTION_V4;
	else
		type = imageNet::CUSTOM;

	return type;
}


// NetworkTypeToStr
const char* imageNet::NetworkTypeToStr( imageNet::NetworkType network )
{
	switch(network)
	{
		case imageNet::ALEXNET:		return "AlexNet";
		case imageNet::GOOGLENET:	return "GoogleNet";
		case imageNet::GOOGLENET_12:	return "GoogleNet-12";
		case imageNet::RESNET_18:	return "ResNet-18";
		case imageNet::RESNET_50:	return "ResNet-50";
		case imageNet::RESNET_101:	return "ResNet-101";
		case imageNet::RESNET_152:	return "ResNet-152";
		case imageNet::VGG_16:		return "VGG-16";
		case imageNet::VGG_19:		return "VGG-19";
		case imageNet::INCEPTION_V4:	return "Inception-v4";
	}

	return "Custom";
}


// Create
imageNet* imageNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("network");
	
	if( !modelName )
		modelName = cmdLine.GetString("model", "googlenet");

	/*if( !modelName )
	{
		if( argc == 2 )
			modelName = argv[1];
		else if( argc == 4 )
			modelName = argv[3];
		else
			modelName = "googlenet";
	}*/

	//if( argc > 3 )
	//	modelName = argv[3];	

	const imageNet::NetworkType type = NetworkTypeFromStr(modelName);

	if( type == imageNet::CUSTOM )
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

		return imageNet::Create(prototxt, modelName, NULL, labels, input, output, maxBatchSize);
	}

	// create from pretrained model
	return imageNet::Create(type);
}


// LoadClassInfo
bool imageNet::LoadClassInfo( const char* filename, std::vector<std::string>& descriptions, std::vector<std::string>& synsets )
{
	if( !filename )
		return false;
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		printf("imageNet -- failed to find %s\n", filename);
		return false;
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		printf("imageNet -- failed to open %s\n", path.c_str());
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
	
	printf("imageNet -- loaded %zu class info entries\n", synsets.size());
	
	if( synsets.size() == 0 )
		return false;

	return true;
}


// LoadClassInfo
bool imageNet::LoadClassInfo( const char* filename, std::vector<std::string>& descriptions )
{
	std::vector<std::string> synsets;
	return LoadClassInfo(filename, descriptions, synsets);
}

	 
// loadClassInfo
bool imageNet::loadClassInfo( const char* filename )
{
	if( !LoadClassInfo(filename, mClassDesc, mClassSynset) )
		return false;

	mClassPath = filename;	
	return true;
}


// PreProcess
bool imageNet::PreProcess( float* rgba, uint32_t width, uint32_t height )
{
	// verify parameters
	if( !rgba || width == 0 || height == 0 )
	{
		printf(LOG_TRT "imageNet::PreProcess( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	if( mNetworkType == imageNet::INCEPTION_V4 )
	{
		// downsample, convert to band-sequential RGB, and apply pixel normalization
		if( CUDA_FAILED(cudaPreImageNetNormRGB((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, make_float2(-1.0f, 1.0f), GetStream())) )
		{
			printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetMean() failed\n");
			return false;
		}
	}
	else
	{
		// downsample, convert to band-sequential BGR, and apply mean pixel subtraction 
		if( CUDA_FAILED(cudaPreImageNetMeanBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									    make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f),
									    GetStream())) )
		{
			printf(LOG_TRT "imageNet::PreProcess() -- cudaPreImageNetMean() failed\n");
			return false;
		}
	}

	PROFILER_END(PROFILER_PREPROCESS);
	return true;
}


// Process
bool imageNet::Process()
{
	void* bindBuffers[] = { mInputCUDA, mOutputs[0].CUDA };	
	cudaStream_t stream = GetStream();

	PROFILER_BEGIN(PROFILER_NETWORK);

	if( !stream )
	{
		//const timespec cpu_begin = timestamp();

	#if 1
		if( !mContext->execute(1, bindBuffers) )
		{
			printf(LOG_TRT "imageNet::Process() -- failed to execute TensorRT network\n");
			return false;
		}
	#else
		const bool result = mContext->enqueue(1, bindBuffers, NULL, NULL);

		CUDA(cudaDeviceSynchronize());

		if( !result )
		{
			printf(LOG_TRT "imageNet::Process() -- failed to enqueue TensorRT network\n");
			return false;
		}
	#endif	

		/*const timespec cpu_end  = timestamp();
		const timespec cpu_time = timeDiff(cpu_begin, cpu_end);

		timePrint(cpu_begin, "imageNet IExecutionContext begin (CPU)  ");
		timePrint(cpu_end, "imageNet IExecutionContext end (CPU)      ");
		timePrint(cpu_time, "imageNet IExecutionContext elapsed (CPU) ");

		printf(LOG_TRT "imageNet IExecutionContext elapsed (CPU):  %fms\n", timeFloat(cpu_time));*/
	}
	else
	{
		//printf("%s stream %p\n", deviceTypeToStr(GetDevice()), GetStream());

		//CUDA(cudaEventRecord(mEvents[0], stream));
		
		// queue the inference processing kernels
		const bool result = mContext->enqueue(1, bindBuffers, stream, NULL);

		//CUDA(cudaEventRecord(mEvents[1], stream));
		//CUDA(cudaEventSynchronize(mEvents[1]));
		CUDA(cudaStreamSynchronize(stream));

		if( !result )
		{
			printf(LOG_TRT "imageNet::Process() -- failed to enqueue TensorRT network\n");
			return false;
		}	
	}

	PROFILER_END(PROFILER_NETWORK);
	return true;
}

				
// Classify
int imageNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	// verify parameters
	if( !rgba || width == 0 || height == 0 )
	{
		printf(LOG_TRT "imageNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}
	
	// downsample and convert to band-sequential BGR
	if( !PreProcess(rgba, width, height) )
	{
		printf(LOG_TRT "imageNet::Classify() -- PreProcess() failed\n");
		return -1;
	}
	
	return Classify(confidence);
}


// Classify
int imageNet::Classify( float* confidence )
{	
	// process with TRT
	if( !Process() )
	{
		printf(LOG_TRT "imageNet::Process() failed\n");
		return -1;
	}
	
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	// determine the maximum class
	int classIndex = -1;
	float classMax = -1.0f;
	
	for( size_t n=0; n < mOutputClasses; n++ )
	{
		const float value = mOutputs[0].CPU[n];
		
		if( value >= 0.01f )
			printf("class %04zu - %f  (%s)\n", n, value, mClassDesc[n].c_str());
	
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

