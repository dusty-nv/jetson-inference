/*
 * inference-101
 */
 
#include "imageNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include <sstream>


// stuff we know about the network and the caffe input/output blobs
static const int MAX_BATCH_SIZE = 1;

const char* INPUT_BLOB_NAME  = "data";
const char* OUTPUT_BLOB_NAME = "prob";



imageNet::imageNet()
{
	mEngine  = NULL;
	mInfer   = NULL;
	mContext = NULL;
	
	mWidth     = 0;
	mHeight    = 0;
	mInputSize = 0;
	mInputCPU  = NULL;
	mInputCUDA = NULL;
	
	mOutputSize    = 0;
	mOutputClasses = 0;
	mOutputCPU     = NULL;
	mOutputCUDA    = NULL;
}


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
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);
	mNetworkType = networkType;

	if( !caffeToGIEModel(proto_file[networkType], model_file[networkType], std::vector< std::string > { OUTPUT_BLOB_NAME }, MAX_BATCH_SIZE, gieModelStream) )
	{
		printf("failed to load %s\n", GetNetworkName());
		return 0;
	}

	printf(LOG_GIE "%s loaded\n", GetNetworkName());
	

	
	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = createInferRuntime(gLogger);
	
	if( !infer )
	{
		printf(LOG_GIE "failed to create InferRuntime\n");
		return 0;
	}
	
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);

	if( !engine )
	{
		printf(LOG_GIE "failed to create CUDA engine\n");
		return 0;
	}
	
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		printf(LOG_GIE "failed to create execution context\n");
		return 0;
	}

	printf(LOG_GIE "CUDA engine context initialized with %u bindings\n", engine->getNbBindings());
	
	mInfer   = infer;
	mEngine  = engine;
	mContext = context;
	
	
	/*
	 * determine dimensions of network bindings
	 */
	const int inputIndex  = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

	printf(LOG_GIE "%s input  binding index:  %i\n", GetNetworkName(), inputIndex);
	printf(LOG_GIE "%s output binding index:  %i\n", GetNetworkName(), outputIndex);
	
	nvinfer1::Dims3 inputDims  = engine->getBindingDimensions(inputIndex);
	nvinfer1::Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	
	size_t inputSize  = inputDims.c * inputDims.h * inputDims.w * sizeof(float);
	size_t outputSize = outputDims.c * outputDims.h * outputDims.w * sizeof(float);
	
	printf(LOG_GIE "%s input  dims (c=%u h=%u w=%u) size=%zu\n", GetNetworkName(), inputDims.c, inputDims.h, inputDims.w, inputSize);
	printf(LOG_GIE "%s output dims (c=%u h=%u w=%u) size=%zu\n", GetNetworkName(), outputDims.c, outputDims.h, outputDims.w, outputSize);
	
	
	/*
	 * allocate memory to hold the input image
	 */
	if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		printf("failed to alloc CUDA mapped memory for imageNet input, %zu bytes\n", inputSize);
		return false;
	}
	
	mInputSize   = inputSize;
	mWidth       = inputDims.w;
	mHeight      = inputDims.h;

	
	/*
	 * allocate output memory to hold the image classes
	 */
	if( !cudaAllocMapped((void**)&mOutputCPU, (void**)&mOutputCUDA, outputSize) )
	{
		printf("failed to alloc CUDA mapped memory for %u output classes\n", outputDims.c);
		return false;
	}
	
	mOutputSize    = outputSize;
	mOutputClasses = outputDims.c;
	
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

