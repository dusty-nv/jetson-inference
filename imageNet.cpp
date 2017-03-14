/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "imageNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"


// constructor
imageNet::imageNet() : tensorNet()
{
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
	
bool imageNet::init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize )
{
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
	mOutputClasses = mOutputs[0].dims.c;
	
	if( !loadClassInfo(class_path) || mClassSynset.size() != mOutputClasses || mClassDesc.size() != mOutputClasses )
	{
		printf("imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mOutputClasses);
		return false;
	}
	
	printf("%s initialized.\n", model_path);
	return true;
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
		
		if( len < syn + 1 )
			continue;
		
		str[syn]   = 0;
		str[len-1] = 0;
		
		const std::string a = str;
		const std::string b = (str + syn + 1);
		
		//printf("a=%s b=%s\n", a.c_str(), b.c_str());
		mClassSynset.push_back(a);
		mClassDesc.push_back(b);
	}
	
	fclose(f);
	
	printf("imageNet -- loaded %zu class info entries\n", mClassSynset.size());
	
	if( mClassSynset.size() == 0 )
		return false;
	
	return true;
}


// init
bool imageNet::init( imageNet::NetworkType networkType, uint32_t maxBatchSize )
{
	const char* proto_file[] = { "alexnet.prototxt", "googlenet.prototxt" };
	const char* model_file[] = { "bvlc_alexnet.caffemodel", "bvlc_googlenet.caffemodel" };

	/*
	 * load and parse googlenet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( proto_file[networkType], model_file[networkType], NULL, "data", "prob", maxBatchSize) )
	{
		printf("failed to load %s\n", model_file[networkType]);
		return false;
	}

	mNetworkType = networkType;
	printf(LOG_GIE "%s loaded\n", GetNetworkName());

	/*
	 * load synset classnames
	 */
	mOutputClasses = mOutputs[0].dims.c;
	
	if( !loadClassInfo("ilsvrc12_synset_words.txt") || mClassSynset.size() != mOutputClasses || mClassDesc.size() != mOutputClasses )
	{
		printf("imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mOutputClasses);
		return false;
	}
	
	printf("%s initialized.\n", GetNetworkName());
	return true;
}


// from imageNet.cu
cudaError_t cudaPreImageNetMean( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );
					
					
// Classify
int imageNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("imageNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNetMean((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
								 make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
	{
		printf("imageNet::Classify() -- cudaPreImageNetMean failed\n");
		return -1;
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

