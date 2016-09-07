/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "imageNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include <sstream>


// stuff we know about the network and the caffe input/output blobs
static const int MAX_BATCH_SIZE = 1;

const char* INPUT_BLOB_NAME  = "data";
const char* OUTPUT_BLOB_NAME = "prob";



// constructor
imageNet::imageNet()
{
	/*mEngine  = NULL;
	mInfer   = NULL;
	mContext = NULL;
	
	mWidth     = 0;
	mHeight    = 0;
	mInputSize = 0;
	mInputCPU  = NULL;
	mInputCUDA = NULL;
	
	mOutputSize    = 0;
	mOutputCPU     = NULL;
	mOutputCUDA    = NULL;*/
	mOutputClasses = 0;
}


// destructor
imageNet::~imageNet()
{
	if( mEngine != NULL )
	{
		mEngine->destroy();
		mEngine = NULL;
	}
		
	if( mInfer != NULL )
	{
		mInfer->destroy();
		mInfer = NULL;
	}
}


// Create
imageNet* imageNet::Create( imageNet::NetworkType networkType )
{
	imageNet* net = new imageNet();
	
	if( !net )
		return NULL;
	
	if( !net->init(networkType) )
	{
		printf("imageNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
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
bool imageNet::init( imageNet::NetworkType networkType )
{
	const char* proto_file[] = { "alexnet.prototxt", "googlenet.prototxt" };
	const char* model_file[] = { "bvlc_alexnet.caffemodel", "bvlc_googlenet.caffemodel" };

	/*
	 * load and parse googlenet network definition and model file
	 */
	if( !tensorNet::LoadNetwork( proto_file[networkType], model_file[networkType], NULL, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME ) )
	{
		printf("failed to load %s\n", model_file[networkType]);
		return false;
	}

	mNetworkType = networkType;
	printf(LOG_GIE "%s loaded\n", GetNetworkName());

	/*
	 * load synset classnames
	 */
	mOutputClasses = mOutputDims.c;
	
	if( !loadClassInfo("ilsvrc12_synset_words.txt") || mClassSynset.size() != mOutputClasses || mClassDesc.size() != mOutputClasses )
	{
		printf("imageNet -- failed to load synset class descriptions  (%zu / %zu of %u)\n", mClassSynset.size(), mClassDesc.size(), mOutputClasses);
		return false;
	}
	
	printf("%s initialized.\n", GetNetworkName());
	return true;
}


// from imageNet.cu
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );
					
					
// Classify
int imageNet::Classify( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("imageNet::Classify( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
	{
		printf("imageNet::Classify() -- cudaPreImageNet failed\n");
		return -1;
	}
	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputCUDA };
	
	mContext->execute(1, inferenceBuffers);
	
	//CUDA(cudaDeviceSynchronize());
	
	// determine the maximum class
	int classIndex = -1;
	float classMax = -1.0f;
	
	for( size_t n=0; n < mOutputClasses; n++ )
	{
		const float value = mOutputCPU[n];
		
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

