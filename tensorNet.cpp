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
 
#include "tensorNet.h"
#include "randInt8Calibrator.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"

#include <iostream>
#include <fstream>
#include <map>


#if NV_TENSORRT_MAJOR > 1
	#define CREATE_INFER_BUILDER nvinfer1::createInferBuilder
	#define CREATE_INFER_RUNTIME nvinfer1::createInferRuntime
#else
	#define CREATE_INFER_BUILDER createInferBuilder
	#define CREATE_INFER_RUNTIME createInferRuntime
#endif


//---------------------------------------------------------------------
const char* precisionTypeToStr( precisionType type )
{
	switch(type)
	{
		case TYPE_DISABLED:	return "DISABLED";
		case TYPE_FASTEST:	return "FASTEST";
		case TYPE_FP32:	return "FP32";
		case TYPE_FP16:	return "FP16";
		case TYPE_INT8:	return "INT8";
	}
}

precisionType precisionTypeFromStr( const char* str )
{
	if( !str )
		return TYPE_DISABLED;

	for( int n=0; n < NUM_PRECISIONS; n++ )
	{
		if( strcasecmp(str, precisionTypeToStr((precisionType)n)) == 0 )
			return (precisionType)n;
	}

	return TYPE_DISABLED;
}

static inline nvinfer1::DataType precisionTypeToTRT( precisionType type )
{
	switch(type)
	{
		case TYPE_FP16:	return nvinfer1::DataType::kHALF;
		case TYPE_INT8:	return nvinfer1::DataType::kINT8;
	}

	return nvinfer1::DataType::kFLOAT;
}

const char* deviceTypeToStr( deviceType type )
{
	switch(type)
	{
		case DEVICE_GPU:	return "GPU";	
		case DEVICE_DLA_0:	return "DLA_0";
		case DEVICE_DLA_1:	return "DLA_1";
	}
}

deviceType deviceTypeFromStr( const char* str )
{
	if( !str )
		return DEVICE_GPU;

 	for( int n=0; n < NUM_DEVICES; n++ )
	{
		if( strcasecmp(str, deviceTypeToStr((deviceType)n)) == 0 )
			return (deviceType)n;
	}

	if( strcasecmp(str, "DLA") == 0 )
		return DEVICE_DLA;

	return DEVICE_GPU;
}

#if NV_TENSORRT_MAJOR >= 5
static inline nvinfer1::DeviceType deviceTypeToTRT( deviceType type )
{
	switch(type)
	{
		case DEVICE_GPU:	return nvinfer1::DeviceType::kGPU;
		//case DEVICE_DLA:	return nvinfer1::DeviceType::kDLA;
#if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0
		case DEVICE_DLA_0:	return nvinfer1::DeviceType::kDLA0;
		case DEVICE_DLA_1:	return nvinfer1::DeviceType::kDLA1;
#else
		case DEVICE_DLA_0:	return nvinfer1::DeviceType::kDLA;
		case DEVICE_DLA_1:	return nvinfer1::DeviceType::kDLA;
#endif
	}
}
#endif
//---------------------------------------------------------------------

// constructor
tensorNet::tensorNet()
{
	mEngine  = NULL;
	mInfer   = NULL;
	mContext = NULL;
	mStream  = NULL;

	mWidth          = 0;
	mHeight         = 0;
	mInputSize      = 0;
	mMaxBatchSize   = 0;
	mInputCPU       = NULL;
	mInputCUDA      = NULL;
	mEnableDebug    = false;
	mEnableProfiler = false;

	mPrecision 	   = TYPE_FASTEST;
	mDevice    	   = DEVICE_GPU;
	mAllowGPUFallback = false;

	memset(mEvents, 0, sizeof(mEvents));

#if NV_TENSORRT_MAJOR < 2
	memset(&mInputDims, 0, sizeof(Dims3));
#endif
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


// EnableProfiler
void tensorNet::EnableProfiler()
{
	mEnableProfiler = true;

	if( mContext != NULL )
		mContext->setProfiler(&gProfiler);
}


// EnableDebug
void tensorNet::EnableDebug()
{
	mEnableDebug = true;
}


// DetectNativePrecisions()
std::vector<precisionType> tensorNet::DetectNativePrecisions( deviceType device )
{
	std::vector<precisionType> types;
	Logger logger;

	// create a temporary builder for querying the supported types
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(logger);
		
	if( !builder )
	{
		printf(LOG_TRT "QueryNativePrecisions() failed to create TensorRT IBuilder instance\n");
		return types;
	}

#if NV_TENSORRT_MAJOR >= 5
	if( device == DEVICE_DLA_0 || device == DEVICE_DLA_1 )
		builder->setFp16Mode(true);

	builder->setDefaultDeviceType( deviceTypeToTRT(device) );
#endif

	// FP32 is supported on all platforms
	types.push_back(TYPE_FP32);

	// detect fast (native) FP16
	if( builder->platformHasFastFp16() )
		types.push_back(TYPE_FP16);

	// detect fast (native) INT8
	if( builder->platformHasFastInt8() )
		types.push_back(TYPE_INT8);

	// print out supported precisions (optional)
	const uint32_t numTypes = types.size();

	printf(LOG_TRT "native precisions detected for %s:  ", deviceTypeToStr(device));
 
	for( uint32_t n=0; n < numTypes; n++ )
	{
		printf("%s", precisionTypeToStr(types[n]));

		if( n < numTypes - 1 )
			printf(", ");
	}

	printf("\n");
	builder->destroy();
	return types;
}


// DetectNativePrecision
bool tensorNet::DetectNativePrecision( const std::vector<precisionType>& types, precisionType type )
{
	const uint32_t numTypes = types.size();

	for( uint32_t n=0; n < numTypes; n++ )
	{
		if( types[n] == type )
			return true;
	}

	return false;
}


// DetectNativePrecision
bool tensorNet::DetectNativePrecision( precisionType precision, deviceType device )
{
	std::vector<precisionType> types = DetectNativePrecisions(device);
	return DetectNativePrecision(types, precision);
}


// FindFastestPrecision
precisionType tensorNet::FindFastestPrecision( deviceType device, bool allowInt8 )
{
	std::vector<precisionType> types = DetectNativePrecisions(device);

	if( allowInt8 && DetectNativePrecision(types, TYPE_INT8) )
		return TYPE_INT8;
	else if( DetectNativePrecision(types, TYPE_FP16) )
		return TYPE_FP16;
	else
		return TYPE_FP32;
}


// Create an optimized GIE network from caffe prototxt and model file
bool tensorNet::ProfileModel(const std::string& deployFile,			   // name for caffe prototxt
					    const std::string& modelFile,			   // name for model 
					    const std::vector<std::string>& outputs,    // network outputs
					    unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
					    precisionType precision, 
					    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, 	
					    std::ostream& gieModelStream)			   // output stream for the GIE model
{
	// create API root class - must span the lifetime of the engine usage
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	builder->setDebugSync(mEnableDebug);
	builder->setMinFindIterations(3);	// allow time for TX1 GPU to spin up
     builder->setAverageFindIterations(2);

	// parse the caffe model to populate the network, then set the outputs
	nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

	//mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
	//printf(LOG_GIE "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
	printf(LOG_GIE "device %s, loading %s %s\n", deviceTypeToStr(device), deployFile.c_str(), modelFile.c_str());
	
	nvinfer1::DataType modelDataType = (precision == TYPE_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // import INT8 weights as FP32
	const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
		parser->parse(deployFile.c_str(),		// caffe deploy file
					  modelFile.c_str(),		// caffe model file
					 *network,					// network definition that the parser will populate
					  modelDataType);

	if( !blobNameToTensor )
	{
		printf(LOG_GIE "device %s, failed to parse caffe network\n", deviceTypeToStr(device));
		return false;
	}
	
	// extract the dimensions of the network input blobs
	std::map<std::string, nvinfer1::Dims3> inputDimensions;

	for( int i=0, n=network->getNbInputs(); i < n; i++ )
	{
		nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
		inputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
		std::cout << LOG_TRT << "retrieved Input tensor \"" << network->getInput(i)->getName() << "\":  " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
	}

	// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
	const size_t num_outputs = outputs.size();
	
	for( size_t n=0; n < num_outputs; n++ )
	{
		nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());
	
		if( !tensor )
			printf(LOG_GIE "failed to retrieve tensor for Output \"%s\"\n", outputs[n].c_str());
		else
		{
			nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(tensor->getDimensions());
			printf(LOG_GIE "retrieved Output tensor \"%s\":  %ix%ix%i\n", tensor->getName(), dims.d[0], dims.d[1], dims.d[2]);
		}

		network->markOutput(*tensor);
	}

	// build the engine
	printf(LOG_GIE "device %s, configuring CUDA engine\n", deviceTypeToStr(device));
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);


	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
		builder->setInt8Mode(true);
		//builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
			calibrator = new randInt8Calibrator(1, mCacheCalibrationPath, inputDimensions);
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
		}

		builder->setInt8Calibrator(calibrator);
	}
	else if( precision == TYPE_FP16 )
	{
		//builder->setHalf2Mode(true);
		builder->setFp16Mode(true);
	}
	

	// set the default device type
#if NV_TENSORRT_MAJOR >= 5
	builder->setDefaultDeviceType(deviceTypeToTRT(device));

	if( allowGPUFallback )
		builder->allowGPUFallback(true);
	
#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
	if( device == DEVICE_DLA_0 )
		builder->setDLACore(0);
	else if( device == DEVICE_DLA_1 )
		builder->setDLACore(1);
#endif
#else
	if( device != DEVICE_GPU )
	{
		printf(LOG_TRT "device %s is not supported in TensorRT %u.%u\n", deviceTypeToStr(device), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif

	// build CUDA engine
	printf(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), builder->getFp16Mode() ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), builder->getInt8Mode() ? "ON" : "OFF"); 
	printf(LOG_GIE "device %s, building CUDA engine\n", deviceTypeToStr(device));

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_GIE "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	printf(LOG_GIE "device %s, completed building CUDA engine\n", deviceTypeToStr(device));

	// we don't need the network definition any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf(LOG_GIE "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	gieModelStream.write((const char*)serMem->data(), serMem->size());
#else
	engine->serialize(gieModelStream);
#endif

	engine->destroy();
	builder->destroy();
	return true;
}


// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, 
					    const char* input_blob, const char* output_blob, uint32_t maxBatchSize,
					    precisionType precision, deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	std::vector<std::string> outputs;
	outputs.push_back(output_blob);
	
	return LoadNetwork(prototxt_path, model_path, mean_path, input_blob, outputs, maxBatchSize, precision, device, allowGPUFallback );
}

		  
// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, 
					    const char* input_blob, const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	if( !prototxt_path || !model_path )
		return false;
	
	printf(LOG_GIE "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
	
	/*
	 * if the precision is left unspecified, detect the fastest
	 */
	printf(LOG_TRT "desired precision specified for %s: %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));

	if( precision == TYPE_DISABLED )
	{
		printf(LOG_TRT "skipping network specified with precision TYPE_DISABLE\n");
		printf(LOG_TRT "please specify a valid precision to create the network\n");

		return false;
	}
	else if( precision == TYPE_FASTEST )
	{
		if( !calibrator )
			printf(LOG_TRT "requested fasted precision for device %s without providing valid calibrator, disabling INT8\n", deviceTypeToStr(device));

		precision = FindFastestPrecision(device, (calibrator != NULL));
		printf(LOG_TRT "selecting fastest native precision for %s:  %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));
	}
	else
	{
		if( !DetectNativePrecision(precision, device) )
		{
			printf(LOG_TRT "precision %s is not supported for device %s\n", precisionTypeToStr(precision), deviceTypeToStr(device));
			return false;
		}

		if( precision == TYPE_INT8 && !calibrator )
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
	}


	/*
	 * attempt to load network from cache before profiling with tensorRT
	 */
	std::stringstream gieModelStream;
	gieModelStream.seekg(0, gieModelStream.beg);

	char cache_prefix[512];
	char cache_path[512];

	sprintf(cache_prefix, "%s.%u.%u.%s.%s", model_path, maxBatchSize, (uint32_t)allowGPUFallback, deviceTypeToStr(device), precisionTypeToStr(precision));
	sprintf(cache_path, "%s.calibration", cache_prefix);
	mCacheCalibrationPath = cache_path;
	
	sprintf(cache_path, "%s.engine", cache_prefix);
	mCacheEnginePath = cache_path;	
	printf(LOG_GIE "attempting to open engine cache file %s\n", mCacheEnginePath.c_str());
	
	std::ifstream cache( mCacheEnginePath );

	if( !cache )
	{
		printf(LOG_GIE "cache file not found, profiling network model on device %s\n", deviceTypeToStr(device));
	
		if( !ProfileModel(prototxt_path, model_path, output_blobs, maxBatchSize, 
					   precision, device, allowGPUFallback, calibrator,
					   gieModelStream) )
		{
			printf("device %s, failed to load %s\n", deviceTypeToStr(device), model_path);
			return 0;
		}
	
		printf(LOG_GIE "network profiling complete, writing engine cache to %s\n", mCacheEnginePath.c_str());
		std::ofstream outFile;
		outFile.open(mCacheEnginePath);
		outFile << gieModelStream.rdbuf();
		outFile.close();
		gieModelStream.seekg(0, gieModelStream.beg);
		printf(LOG_GIE "device %s, completed writing engine cache to %s\n", deviceTypeToStr(device), mCacheEnginePath.c_str());
	}
	else
	{
		printf(LOG_GIE "loading network profile from engine cache... %s\n", mCacheEnginePath.c_str());
		gieModelStream << cache.rdbuf();
		cache.close();

		// test for half FP16 support
		/*nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
		
		if( builder != NULL )
		{
			mEnableFP16 = !mOverride16 && builder->platformHasFastFp16();
			printf(LOG_GIE "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
			builder->destroy();	
		}*/
	}

	printf(LOG_GIE "device %s, %s loaded\n", deviceTypeToStr(device), model_path);
	

	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);
	
	if( !infer )
	{
		printf(LOG_GIE "device %s, failed to create InferRuntime\n", deviceTypeToStr(device));
		return 0;
	}

#if NV_TENSORRT_MAJOR >= 5 && !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
	// if using DLA, set the desired core before deserialization occurs
	if( device == DEVICE_DLA_0 )
	{
		printf(LOG_TRT "device %s, enabling DLA core 0\n", deviceTypeToStr(device));
		infer->setDLACore(0);
	}
	else if( device == DEVICE_DLA_1 )
	{
		printf(LOG_TRT "device %s, enabling DLA core 1\n", deviceTypeToStr(device));
		infer->setDLACore(1);
	}
#endif

#if NV_TENSORRT_MAJOR > 1
	// support for stringstream deserialization was deprecated in TensorRT v2
	// instead, read the stringstream into a memory buffer and pass that to TRT.
	gieModelStream.seekg(0, std::ios::end);
	const int modelSize = gieModelStream.tellg();
	gieModelStream.seekg(0, std::ios::beg);

	void* modelMem = malloc(modelSize);

	if( !modelMem )
	{
		printf(LOG_GIE "failed to allocate %i bytes to deserialize model\n", modelSize);
		return 0;
	}

	gieModelStream.read((char*)modelMem, modelSize);
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(modelMem, modelSize, NULL);
	free(modelMem);
#else
	// TensorRT v1 can deserialize directly from stringstream
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream);
#endif

	if( !engine )
	{
		printf(LOG_GIE "device %s, failed to create CUDA engine\n", deviceTypeToStr(device));
		return 0;
	}
	
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		printf(LOG_GIE "device %s, failed to create execution context\n", deviceTypeToStr(device));
		return 0;
	}

	if( mEnableDebug )
	{
		printf(LOG_GIE "device %s, enabling context debug sync.\n", deviceTypeToStr(device));
		context->setDebugSync(true);
	}

	if( mEnableProfiler )
		context->setProfiler(&gProfiler);

	printf(LOG_GIE "device %s, CUDA engine context initialized with %u bindings\n", deviceTypeToStr(device), engine->getNbBindings());
	
	mInfer   = infer;
	mEngine  = engine;
	mContext = context;
	
	SetStream(stream);	// set default device stream

	
	/*
	 * determine dimensions of network input bindings
	 */
	const int inputIndex = engine->getBindingIndex(input_blob);
	
	printf(LOG_GIE "%s input  binding index:  %i\n", model_path, inputIndex);
	
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::Dims inputDims = engine->getBindingDimensions(inputIndex);
#else
	Dims3 inputDims = engine->getBindingDimensions(inputIndex);
#endif

	size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);

	printf(LOG_GIE "%s input  dims (b=%u c=%u h=%u w=%u) size=%zu\n", model_path, maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);
	

	/*
	 * allocate memory to hold the input image
	 */
	//if( CUDA_FAILED(cudaMalloc((void**)&mInputCUDA, inputSize)) )
	if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		printf("failed to alloc CUDA mapped memory for tensorNet input, %zu bytes\n", inputSize);
		return false;
	}
	
	mInputSize    = inputSize;
	mWidth        = DIMS_W(inputDims);
	mHeight       = DIMS_H(inputDims);
	mMaxBatchSize = maxBatchSize;
	

	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();
	
	for( int n=0; n < numOutputs; n++ )
	{
		const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());
		printf(LOG_GIE "%s output %i %s  binding index:  %i\n", model_path, n, output_blobs[n].c_str(), outputIndex);

	#if NV_TENSORRT_MAJOR > 1
		nvinfer1::Dims outputDims = engine->getBindingDimensions(outputIndex);
	#else
		Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	#endif

		size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
		printf(LOG_GIE "%s output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", model_path, n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		//if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			printf("failed to alloc CUDA mapped memory for %u output classes\n", DIMS_C(outputDims));
			return false;
		}
	
		outputLayer l;
		
		l.CPU  = (float*)outputCPU;
		l.CUDA = (float*)outputCUDA;
		l.size = outputSize;

	#if NV_TENSORRT_MAJOR > 1
		DIMS_W(l.dims) = DIMS_W(outputDims);
		DIMS_H(l.dims) = DIMS_H(outputDims);
		DIMS_C(l.dims) = DIMS_C(outputDims);
	#else
		l.dims = outputDims;
	#endif

		l.name = output_blobs[n];
		mOutputs.push_back(l);
	}
	

#if NV_TENSORRT_MAJOR > 1
	DIMS_W(mInputDims) = DIMS_W(inputDims);
	DIMS_H(mInputDims) = DIMS_H(inputDims);
	DIMS_C(mInputDims) = DIMS_C(inputDims);
#else
	mInputDims        = inputDims;
#endif
	mPrototxtPath     = prototxt_path;
	mModelPath        = model_path;
	mInputBlobName    = input_blob;
	mPrecision        = precision;
	mDevice           = device;
	mAllowGPUFallback = allowGPUFallback;

	if( mean_path != NULL )
		mMeanPath = mean_path;
	
	printf("device %s, %s initialized.\n", deviceTypeToStr(device), mModelPath.c_str());
	return true;
}


// CreateStream
cudaStream_t tensorNet::CreateStream( bool nonBlocking )
{
	uint32_t flags = cudaStreamDefault;

	if( nonBlocking )
		flags = cudaStreamNonBlocking;

	cudaStream_t stream = NULL;

	if( CUDA_FAILED(cudaStreamCreateWithFlags(&stream, flags)) )
		return NULL;

	SetStream(stream);
	return stream;
}


// SetStream
void tensorNet::SetStream( cudaStream_t stream )
{
	mStream = stream;

	if( !mStream )
		return;

	for( int n=0; n < 2; n++ )
	{
		if( !mEvents[n] )
			CUDA(cudaEventCreateWithFlags(&mEvents[n], /*cudaEventBlockingSync*/cudaEventDefault));
	}
}	

