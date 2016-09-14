/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "tensorNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"


static const int MAX_BATCH_SIZE = 1;


// constructor
tensorNet::tensorNet()
{
	mEngine  = NULL;
	mInfer   = NULL;
	mContext = NULL;
	
	mWidth     = 0;
	mHeight    = 0;
	mInputSize = 0;
	mInputCPU  = NULL;
	mInputCUDA = NULL;
	
	memset(&mInputDims, 0, sizeof(nvinfer1::Dims3));
}


// Destructor
tensorNet::~tensorNet()
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


// Create an optimized GIE network from caffe prototxt and model file
bool tensorNet::ProfileModel(const std::string& deployFile,			   // name for caffe prototxt
					         const std::string& modelFile,			   // name for model 
					         const std::vector<std::string>& outputs,   // network outputs
					         unsigned int maxBatchSize,				   // batch size - NB must be at least as large as the batch we want to run with)
					         std::ostream& gieModelStream)			   // output stream for the GIE model
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
    builder->setAverageFindIterations(2);

	// parse the caffe model to populate the network, then set the outputs
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	const bool useFp16 = builder->platformHasFastFp16();	// getHalf2Mode();
	printf(LOG_GIE "platform %s FP16 support.\n", useFp16 ? "has" : "does not have");
	printf(LOG_GIE "loading %s %s\n", deployFile.c_str(), modelFile.c_str());
	
	nvinfer1::DataType modelDataType = useFp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // create a 16-bit model if it's natively supported
	const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
		parser->parse(deployFile.c_str(),		// caffe deploy file
					  modelFile.c_str(),		// caffe model file
					 *network,					// network definition that the parser will populate
					  modelDataType);

	if( !blobNameToTensor )
	{
		printf(LOG_GIE "failed to parse caffe network\n");
		return false;
	}
	
	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	const size_t num_outputs = outputs.size();
	
	for( size_t n=0; n < num_outputs; n++ )
		network->markOutput(*blobNameToTensor->find(outputs[n].c_str()));


	// Build the engine
	printf(LOG_GIE "configuring CUDA engine\n");
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	// set up the network for paired-fp16 format, only on DriveCX
	if(useFp16)
		builder->setHalf2Mode(true);

	printf(LOG_GIE "building CUDA engine\n");
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_GIE "failed to build CUDA engine\n");
		return false;
	}

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy(); //delete parser;

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	
	return true;
}


// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, const char* input_blob, const char* output_blob)
{
	std::vector<std::string> outputs;
	outputs.push_back(output_blob);
	
	return LoadNetwork(prototxt_path, model_path, mean_path, input_blob, outputs);
}

				  
// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, const char* input_blob, const std::vector<std::string>& output_blobs)
{
	if( !prototxt_path || !model_path )
		return false;
	
	/*
	 * load and parse network definition and model file
	 */
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	if( !ProfileModel(prototxt_path, model_path, output_blobs, MAX_BATCH_SIZE, gieModelStream) )
	{
		printf("failed to load %s\n", model_path);
		return 0;
	}

	printf(LOG_GIE "%s loaded\n", model_path);
	

	
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
	 * determine dimensions of network input bindings
	 */
	const int inputIndex = engine->getBindingIndex(input_blob);
	
	printf(LOG_GIE "%s input  binding index:  %i\n", model_path, inputIndex);
	
	nvinfer1::Dims3 inputDims  = engine->getBindingDimensions(inputIndex);
	size_t inputSize  = inputDims.c * inputDims.h * inputDims.w * sizeof(float);
	
	printf(LOG_GIE "%s input  dims (c=%u h=%u w=%u) size=%zu\n", model_path, inputDims.c, inputDims.h, inputDims.w, inputSize);
	
	/*
	 * allocate memory to hold the input image
	 */
	if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		printf("failed to alloc CUDA mapped memory for tensorNet input, %zu bytes\n", inputSize);
		return false;
	}
	
	mInputSize   = inputSize;
	mWidth       = inputDims.w;
	mHeight      = inputDims.h;
	
	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();
	
	for( int n=0; n < numOutputs; n++ )
	{
		const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
		printf(LOG_GIE "%s output %i %s  binding index:  %i\n", model_path, n, output_blobs[n].c_str(), outputIndex);
		nvinfer1::Dims3 outputDims = engine->getBindingDimensions(outputIndex);
		size_t outputSize = outputDims.c * outputDims.h * outputDims.w * sizeof(float);
		printf(LOG_GIE "%s output %i %s  dims (c=%u h=%u w=%u) size=%zu\n", model_path, n, output_blobs[n].c_str(), outputDims.c, outputDims.h, outputDims.w, outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			printf("failed to alloc CUDA mapped memory for %u output classes\n", outputDims.c);
			return false;
		}
	
		outputLayer l;
		
		l.CPU  = (float*)outputCPU;
		l.CUDA = (float*)outputCUDA;
		l.size = outputSize;
		l.dims = outputDims;
		l.name = output_blobs[n];
		
		mOutputs.push_back(l);
	}
	

	mInputDims      = inputDims;
	mPrototxtPath   = prototxt_path;
	mModelPath      = model_path;
	mMeanPath       = mean_path;
	mInputBlobName  = input_blob;

	printf("%s initialized.\n", mModelPath.c_str());
	return true;
}

