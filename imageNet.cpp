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
#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "commandLine.h"

// constructor
imageNet::imageNet() : tensorNet()
{
	mCustomClasses = 0;
	mOutputClasses = 0;
}


// destructor
imageNet::~imageNet()
{

}


// Create
imageNet* imageNet::Create( imageNet::NetworkType networkType, uint32_t maxBatchSize )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(networkType, maxBatchSize) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// Create
imageNet* imageNet::Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
							const char* class_path, const char* input, const char* output, uint32_t maxBatchSize )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(prototxt_path, model_path, mean_binary, class_path, input, output, maxBatchSize) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


// init
bool imageNet::init( imageNet::NetworkType networkType, uint32_t maxBatchSize )
{
	/*const char* proto_file[] = { "networks/alexnet.prototxt", "networks/googlenet.prototxt" };
	const char* model_file[] = { "networks/bvlc_alexnet.caffemodel", "networks/bvlc_googlenet.caffemodel" };

	if( !tensorNet::LoadNetwork( proto_file[networkType], model_file[networkType], NULL, "data", "prob", maxBatchSize) )
	{
		printf("failed to load %s\n", model_file[networkType]);
		return false;
	}

	mNetworkType = networkType;
	printf(LOG_GIE "%s loaded\n", GetNetworkName());


	mOutputClasses = mOutputs[0].dims.c;
	
	if( !loadClassInfo("networks/ilsvrc12_synset_words.txt") || mClassSynset.size() != mOutputClasses || mClassDesc.size() != mOutputClasses )
	{
		printf("imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mOutputClasses);
		return false;
	}
	
	printf("%s initialized.\n", GetNetworkName());
	return true;*/

	if( networkType == imageNet::ALEXNET )
		return init( "networks/alexnet.prototxt", "networks/bvlc_alexnet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize );
	else if( networkType == imageNet::GOOGLENET )
		return init( "networks/googlenet.prototxt", "networks/bvlc_googlenet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize );
	else if( networkType == imageNet::GOOGLENET_12 )
		return init( "networks/GoogleNet-ILSVRC12-subset/deploy.prototxt", "networks/GoogleNet-ILSVRC12-subset/snapshot_iter_184080.caffemodel", NULL, "networks/GoogleNet-ILSVRC12-subset/labels.txt", IMAGENET_DEFAULT_INPUT, "softmax", maxBatchSize );
	else
		return init( "networks/googlenet.prototxt", "networks/bvlc_googlenet.caffemodel", NULL, "networks/ilsvrc12_synset_words.txt", IMAGENET_DEFAULT_INPUT, IMAGENET_DEFAULT_OUTPUT, maxBatchSize );
}


// init
bool imageNet::init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize )
{
	if( !prototxt_path || !model_path || !class_path || !input || !output )
		return false;

	printf("\n");
	printf("imageNet -- loading classification network model from:\n");
	printf("         -- prototxt     %s\n", prototxt_path);
	printf("         -- model        %s\n", model_path);
	printf("         -- mean_binary  %s\n", mean_binary);
	printf("         -- class_labels %s\n", class_path);
	printf("         -- input_blob   '%s'\n", input);
	printf("         -- output_blob  '%s'\n", output);
	printf("         -- batch_size   %u\n\n", maxBatchSize);

	/*
	 * load and parse googlenet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( prototxt_path, model_path, mean_binary, input, output, maxBatchSize ) )
	{
		printf("failed to load %s\n", model_path);
		return false;
	}

	printf(LOG_GIE "%s loaded\n", model_path);

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
			


// Create
imageNet* imageNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* modelName = cmdLine.GetString("model");

	if( !modelName )
	{
		if( argc == 2 )
			modelName = argv[1];
		else if( argc == 4 )
			modelName = argv[3];
		else
			modelName = "googlenet";
	}

	//if( argc > 3 )
	//	modelName = argv[3];	

	imageNet::NetworkType type = imageNet::GOOGLENET;

	if( strcasecmp(modelName, "alexnet") == 0 )
	{
		type = imageNet::ALEXNET;
	}
	else if( strcasecmp(modelName, "googlenet") == 0 )
	{
		type = imageNet::GOOGLENET;
	}
	else if( strcasecmp(modelName, "googlenet-12") == 0 || strcasecmp(modelName, "googlenet_12") == 0 )
	{
		type = imageNet::GOOGLENET_12;
	}
	else
	{
		const char* prototxt = cmdLine.GetString("prototxt");
		const char* labels   = cmdLine.GetString("labels");
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");
		const char* out_bbox = cmdLine.GetString("output_bbox");
		const char* mean     = cmdLine.GetString("mean_binary");
		
		if( !input ) 	input    = IMAGENET_DEFAULT_INPUT;
		if( !output )  output   = IMAGENET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = 2;

		return imageNet::Create(prototxt, modelName, mean, labels, input, output, maxBatchSize);
	}

	// create from pretrained model
	return imageNet::Create(type);
}
				 

// loadClassInfo
bool imageNet::loadClassInfo( const char* filename )
{
	if( !filename )
		return false;
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("imageNet -- failed to open %s\n", filename);
		return false;
	}
	
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
	
	printf("imageNet -- loaded %zu class info entries\n", mClassSynset.size());
	
	if( mClassSynset.size() == 0 )
		return false;
	
	return true;
}



// from imageNet.cu
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, float* mean_binary );
					
// Classify
int imageNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("imageNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	if( !mMeanCPU )
	{
		// downsample and convert to band-sequential BGR
		if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									 make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
		{
			printf("imageNet::Classify() -- cudaPreImageNetMean failed\n");
			return -1;
		}
	} else {
		if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									 mMeanCUDA)) )
		{
			printf("imageNet::Classify() -- cudaPreImageNetMean failed\n");
			return -1;
		}
	}
	
	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	mContext->execute(1, inferenceBuffers);
	
	//CUDA(cudaDeviceSynchronize());
	PROFILER_REPORT();
	
	
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
	return classIndex;
}

