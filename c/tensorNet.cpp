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
#include "filesystem.h"

#include "NvCaffeParser.h"

#if NV_TENSORRT_MAJOR >= 5
#include "NvOnnxParser.h"
#include "NvUffParser.h"
#include "NvInferPlugin.h"
#endif

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

#define LOG_DOWNLOADER_TOOL "        if loading a built-in model, maybe it wasn't downloaded before.\n\n"    \
					   "        Run the Model Downloader tool again and select it for download:\n\n"   \
					   "           $ cd <jetson-inference>/tools\n" 	  	\
					   "           $ ./download-models.sh\n"


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
#if NV_TENSORRT_MAJOR >= 4
		case TYPE_INT8:	return nvinfer1::DataType::kINT8;
#endif
	}

	return nvinfer1::DataType::kFLOAT;
}

static inline bool isFp16Enabled( nvinfer1::IBuilder* builder )
{
#if NV_TENSORRT_MAJOR < 4
	return builder->getHalf2Mode();
#else
	return builder->getFp16Mode();
#endif
}

static inline bool isInt8Enabled( nvinfer1::IBuilder* builder )
{
#if NV_TENSORRT_MAJOR >= 4
	return builder->getInt8Mode();
#else
	return false;
#endif
}

#if NV_TENSORRT_MAJOR >= 4
static inline const char* dataTypeToStr( nvinfer1::DataType type )
{
	switch(type)
	{
		case nvinfer1::DataType::kFLOAT:	return "FP32";
		case nvinfer1::DataType::kHALF:	return "FP16";
		case nvinfer1::DataType::kINT8:	return "INT8";
		case nvinfer1::DataType::kINT32:	return "INT32";
	}

	printf(LOG_TRT "warning -- unknown nvinfer1::DataType (%i)\n", (int)type);
	return "UNKNOWN";
}

static inline const char* dimensionTypeToStr( nvinfer1::DimensionType type )
{
	switch(type)
	{
		case nvinfer1::DimensionType::kSPATIAL:	 return "SPATIAL";
		case nvinfer1::DimensionType::kCHANNEL:	 return "CHANNEL";
		case nvinfer1::DimensionType::kINDEX:	 return "INDEX";
		case nvinfer1::DimensionType::kSEQUENCE: return "SEQUENCE";
	}

	printf(LOG_TRT "warning -- unknown nvinfer1::DimensionType (%i)\n", (int)type);
	return "UNKNOWN";
}
#endif

#if NV_TENSORRT_MAJOR > 1
static inline nvinfer1::Dims validateDims( const nvinfer1::Dims& dims )
{
	if( dims.nbDims == nvinfer1::Dims::MAX_DIMS )
		return dims;
	
	nvinfer1::Dims dims_out = dims;

	// TRT doesn't set the higher dims, so make sure they are 1
	for( int n=dims_out.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
		dims_out.d[n] = 1;

	return dims_out;
}
#endif

#if NV_TENSORRT_MAJOR >= 7
static inline nvinfer1::Dims shiftDims( const nvinfer1::Dims& dims )
{
    // TensorRT 7.0 requires EXPLICIT_BATCH flag for ONNX models,
    // which adds a batch dimension (4D NCHW), whereas historically
    // 3D CHW was expected.  Remove the batch dim (it is typically 1)
    nvinfer1::Dims out = dims;

    out.d[0] = dims.d[1];
    out.d[1] = dims.d[2];
    out.d[2] = dims.d[3];
    out.d[3] = 1;

    return out;
}
#endif

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

const char* modelTypeToStr( modelType format )
{
	switch(format)
	{
		case MODEL_CUSTOM:	return "custom";	
		case MODEL_CAFFE:	return "caffe";
		case MODEL_ONNX:	return "ONNX";
		case MODEL_UFF:	return "UFF";
	}
}

modelType modelTypeFromStr( const char* str )
{
	if( !str )
		return MODEL_CUSTOM;

	if( strcasecmp(str, "caffemodel") == 0 || strcasecmp(str, "caffe") == 0 )
		return MODEL_CAFFE;
	else if( strcasecmp(str, "onnx") == 0 )
		return MODEL_ONNX;
	else if( strcasecmp(str, "uff") == 0 )
		return MODEL_UFF;

	return MODEL_CUSTOM;
}

const char* profilerQueryToStr( profilerQuery query )
{
	switch(query)
	{
		case PROFILER_PREPROCESS:  return "Pre-Process";
		case PROFILER_NETWORK:	  return "Network";
		case PROFILER_POSTPROCESS: return "Post-Process";
		case PROFILER_VISUALIZE:	  return "Visualize";
		case PROFILER_TOTAL:	  return "Total";
	}
}

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

	mModelType        = MODEL_CUSTOM;
	mPrecision 	   = TYPE_FASTEST;
	mDevice    	   = DEVICE_GPU;
	mAllowGPUFallback = false;

	mProfilerQueriesUsed = 0;
	mProfilerQueriesDone = 0;

	memset(mEventsCPU, 0, sizeof(mEventsCPU));
	memset(mEventsGPU, 0, sizeof(mEventsGPU));
	memset(mProfilerTimes, 0, sizeof(mProfilerTimes));

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
void tensorNet::EnableLayerProfiler()
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

#if NV_TENSORRT_MAJOR >= 4
	// detect fast (native) INT8
	if( builder->platformHasFastInt8() )
		types.push_back(TYPE_INT8);
#endif

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
					    const char* input, const Dims3& inputDims,
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

	//mEnableFP16 = (mOverride16 == true) ? false : builder->platformHasFastFp16();
	//printf(LOG_TRT "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
	printf(LOG_TRT "device %s, loading %s %s\n", deviceTypeToStr(device), deployFile.c_str(), modelFile.c_str());
	

	// parse the different types of model formats
	if( mModelType == MODEL_CAFFE )
	{
		// parse the caffe model to populate the network, then set the outputs
		nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();

		nvinfer1::DataType modelDataType = (precision == TYPE_FP16) ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT; // import INT8 weights as FP32
		const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
			parser->parse(deployFile.c_str(),		// caffe deploy file
						  modelFile.c_str(),	// caffe model file
						 *network,			// network definition that the parser will populate
						  modelDataType);

		if( !blobNameToTensor )
		{
			printf(LOG_TRT "device %s, failed to parse caffe network\n", deviceTypeToStr(device));
			return false;
		}

		// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
		const size_t num_outputs = outputs.size();
		
		for( size_t n=0; n < num_outputs; n++ )
		{
			nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());
		
			if( !tensor )
				printf(LOG_TRT "failed to retrieve tensor for Output \"%s\"\n", outputs[n].c_str());
			else
			{
			#if NV_TENSORRT_MAJOR >= 4
				nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(tensor->getDimensions());
				printf(LOG_TRT "retrieved Output tensor \"%s\":  %ix%ix%i\n", tensor->getName(), dims.d[0], dims.d[1], dims.d[2]);
			#endif
			}

			network->markOutput(*tensor);
		}

		//parser->destroy();
	}
#if NV_TENSORRT_MAJOR >= 5
	else if( mModelType == MODEL_ONNX )
	{
    #if NV_TENSORRT_MAJOR >= 7
        network->destroy();
        network = builder->createNetworkV2(1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

        if( !network )
        {
            printf(LOG_TRT "IBuilder::createNetworkV2(EXPLICIT_BATCH) failed\n");
            return false;
        }
    #endif

		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

		if( !parser )
		{
			printf(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
			return false;
		}

    #if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kINFO;
    #else
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kVERBOSE;
    #endif

		if( !parser->parseFromFile(modelFile.c_str(), parserLogLevel) )
		{
			printf(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
			return false;
		}

		//parser->destroy();
	}
	else if( mModelType == MODEL_UFF )
	{
		// create parser instance
		nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
		
		if( !parser )
		{
			printf(LOG_TRT "failed to create UFF parser\n");
			return false;
		}
		
		// register input
		if( !parser->registerInput(input, inputDims, nvuffparser::UffInputOrder::kNCHW) )
		{
			printf(LOG_TRT "failed to register input '%s' for UFF model '%s'\n", input, modelFile.c_str());
			return false;
		}
		
		// register outputs
		/*const size_t numOutputs = outputs.size();
		
		for( uint32_t n=0; n < numOutputs; n++ )
		{
			if( !parser->registerOutput(outputs[n].c_str()) )
				printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", outputs[n].c_str(), modelFile.c_str());
		}*/

		if( !parser->registerOutput("MarkOutput_0") )
			printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", "MarkOutput_0", modelFile.c_str());

		
		// parse network
		if( !parser->parse(modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT) )
		{
			printf(LOG_TRT "failed to parse UFF model '%s'\n", modelFile.c_str());
			return false;
		}
		
		//parser->destroy();
	}
#endif

	// build the engine
	printf(LOG_TRT "device %s, configuring CUDA engine\n", deviceTypeToStr(device));
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);


	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
	#if NV_TENSORRT_MAJOR >= 4
		builder->setInt8Mode(true);
		//builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
	        // extract the dimensions of the network input blobs
	        std::map<std::string, nvinfer1::Dims3> inputDimensions;

	        for( int i=0, n=network->getNbInputs(); i < n; i++ )
	        {
                nvinfer1::Dims dims = network->getInput(i)->getDimensions();

            #if NV_TENSORRT_MAJOR >= 7
                if( mModelType == MODEL_ONNX )
                    dims = shiftDims(dims);  // change NCHW to CHW for EXPLICIT_BATCH
            #endif

		        //nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
		        inputDimensions.insert(std::make_pair(network->getInput(i)->getName(), static_cast<nvinfer1::Dims3&&>(dims)));
		        std::cout << LOG_TRT << "retrieved Input tensor \"" << network->getInput(i)->getName() << "\":  " << dims.d[0] << "x" << dims.d[1] << "x" << dims.d[2] << std::endl;
	        }

            // default to random calibration
			calibrator = new randInt8Calibrator(1, mCacheCalibrationPath, inputDimensions);
			printf(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
		}

		builder->setInt8Calibrator(calibrator);
	#else
		printf(LOG_TRT "INT8 precision requested, and TensorRT %u.%u doesn't meet minimum version for INT8\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		printf(LOG_TRT "please use minumum version of TensorRT 4.0 or newer for INT8 support\n");

		return false;
	#endif
	}
	else if( precision == TYPE_FP16 )
	{
	#if NV_TENSORRT_MAJOR < 4
		builder->setHalf2Mode(true);
	#else
		builder->setFp16Mode(true);
	#endif
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
	printf(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), isFp16Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), isInt8Enabled(builder) ? "ON" : "OFF"); 
	printf(LOG_TRT "device %s, building CUDA engine (this may take a few minutes the first time a network is loaded)\n", deviceTypeToStr(device));

	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
	
	if( !engine )
	{
		printf(LOG_TRT "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	printf(LOG_TRT "device %s, completed building CUDA engine\n", deviceTypeToStr(device));

	// we don't need the network definition any more, and we can destroy the parser
	network->destroy();
	//parser->destroy();

	// serialize the engine, then close everything down
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::IHostMemory* serMem = engine->serialize();

	if( !serMem )
	{
		printf(LOG_TRT "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
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
bool tensorNet::LoadNetwork( const char* prototxt_path_, const char* model_path_, const char* mean_path, 
					    const char* input_blob, const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	return LoadNetwork(prototxt_path_, model_path_, mean_path,
					   input_blob, Dims3(1,1,1), output_blobs,
					   maxBatchSize, precision, device,
					   allowGPUFallback, calibrator, stream);
}

					   
// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path_, const char* model_path_, const char* mean_path, 
					    const char* input_blob, const Dims3& input_dims,
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	if( /*!prototxt_path_ ||*/ !model_path_ )
		return false;

#if NV_TENSORRT_MAJOR >= 4
	printf(LOG_TRT "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
#else
	printf(LOG_TRT "TensorRT version %u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
#endif

	/*
	 * load NV inference plugins
	 */
#if NV_TENSORRT_MAJOR > 4
	static bool loadedPlugins = false;

	if( !loadedPlugins )
	{
		printf(LOG_TRT "loading NVIDIA plugins...\n");

		loadedPlugins = initLibNvInferPlugins(&gLogger, "");

		if( !loadedPlugins )
			printf(LOG_TRT "failed to load NVIDIA plugins\n");
		else
			printf(LOG_TRT "completed loading NVIDIA plugins.\n");
	}
#endif

	/*
	 * verify the prototxt and model paths
	 */
	const std::string model_path    = locateFile(model_path_);
	const std::string prototxt_path = locateFile(prototxt_path_ != NULL ? prototxt_path_ : "");
	
	const std::string model_ext = fileExtension(model_path_);
	const modelType   model_fmt = modelTypeFromStr(model_ext.c_str());

	printf(LOG_TRT "detected model format - %s  (extension '.%s')\n", modelTypeToStr(model_fmt), model_ext.c_str());

	if( model_fmt == MODEL_CUSTOM )
	{
		printf(LOG_TRT "model format '%s' not supported by jetson-inference\n", modelTypeToStr(model_fmt));
		return false;
	}
#if NV_TENSORRT_MAJOR < 5
	else if( model_fmt == MODEL_ONNX )
	{
		printf(LOG_TRT "importing ONNX models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
	else if( model_fmt == MODEL_UFF )
	{
		printf(LOG_TRT "importing UFF models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif
	else if( model_fmt == MODEL_CAFFE && !prototxt_path_ )
	{
		printf(LOG_TRT "attempted to load caffe model without specifying prototxt file\n");
		return false;
	}

	mModelType = model_fmt;


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

	sprintf(cache_prefix, "%s.%u.%u.%s.%s", model_path.c_str(), maxBatchSize, (uint32_t)allowGPUFallback, deviceTypeToStr(device), precisionTypeToStr(precision));
	sprintf(cache_path, "%s.calibration", cache_prefix);
	mCacheCalibrationPath = cache_path;
	
	sprintf(cache_path, "%s.engine", cache_prefix);
	mCacheEnginePath = cache_path;	
	printf(LOG_TRT "attempting to open engine cache file %s\n", mCacheEnginePath.c_str());
	
	std::ifstream cache( mCacheEnginePath );

	if( !cache )
	{
		printf(LOG_TRT "cache file not found, profiling network model on device %s\n", deviceTypeToStr(device));
	
		if( model_path.size() == 0 )
		{
			printf("\nerror:  model file '%s' was not found.\n", model_path_);
			printf("%s\n", LOG_DOWNLOADER_TOOL);
			return 0;
		}

		if( !ProfileModel(prototxt_path, model_path, input_blob, input_dims,
						 output_blobs, maxBatchSize, precision, device, 
						 allowGPUFallback, calibrator, gieModelStream) )
		{
			printf(LOG_TRT "device %s, failed to load %s\n", deviceTypeToStr(device), model_path_);
			return 0;
		}
	
		printf(LOG_TRT "network profiling complete, writing engine cache to %s\n", mCacheEnginePath.c_str());
		std::ofstream outFile;
		outFile.open(mCacheEnginePath);
		outFile << gieModelStream.rdbuf();
		outFile.close();
		gieModelStream.seekg(0, gieModelStream.beg);
		printf(LOG_TRT "device %s, completed writing engine cache to %s\n", deviceTypeToStr(device), mCacheEnginePath.c_str());
	}
	else
	{
		printf(LOG_TRT "loading network profile from engine cache... %s\n", mCacheEnginePath.c_str());
		gieModelStream << cache.rdbuf();
		cache.close();

		// test for half FP16 support
		/*nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
		
		if( builder != NULL )
		{
			mEnableFP16 = !mOverride16 && builder->platformHasFastFp16();
			printf(LOG_TRT "platform %s fast FP16 support\n", mEnableFP16 ? "has" : "does not have");
			builder->destroy();	
		}*/
	}

	printf(LOG_TRT "device %s, %s loaded\n", deviceTypeToStr(device), model_path.c_str());
	

	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);
	
	if( !infer )
	{
		printf(LOG_TRT "device %s, failed to create InferRuntime\n", deviceTypeToStr(device));
		return 0;
	}

#if NV_TENSORRT_MAJOR >= 5 
#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
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
		printf(LOG_TRT "failed to allocate %i bytes to deserialize model\n", modelSize);
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
		printf(LOG_TRT "device %s, failed to create CUDA engine\n", deviceTypeToStr(device));
		return 0;
	}
	
	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		printf(LOG_TRT "device %s, failed to create execution context\n", deviceTypeToStr(device));
		return 0;
	}

	if( mEnableDebug )
	{
		printf(LOG_TRT "device %s, enabling context debug sync.\n", deviceTypeToStr(device));
		context->setDebugSync(true);
	}

	if( mEnableProfiler )
		context->setProfiler(&gProfiler);

	printf(LOG_TRT "device %s, CUDA engine context initialized with %u bindings\n", deviceTypeToStr(device), engine->getNbBindings());
	
	mInfer   = infer;
	mEngine  = engine;
	mContext = context;
	
	SetStream(stream);	// set default device stream


#if NV_TENSORRT_MAJOR >= 4
	/*
	 * print out binding info
	 */
	const int numBindings = engine->getNbBindings();
	
	for( int n=0; n < numBindings; n++ )
	{
		printf(LOG_TRT "binding -- index   %i\n", n);

		const char* bind_name = engine->getBindingName(n);

		printf("               -- name    '%s'\n", bind_name);
		printf("               -- type    %s\n", dataTypeToStr(engine->getBindingDataType(n)));
		printf("               -- in/out  %s\n", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");

		const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

		printf("               -- # dims  %i\n", bind_dims.nbDims);
		
		for( int i=0; i < bind_dims.nbDims; i++ )
			printf("               -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));
	}
#endif

	/*
	 * determine dimensions of network input bindings
	 */
	const int inputIndex = engine->getBindingIndex(input_blob);
	
	printf(LOG_TRT "binding to input 0 %s  binding index:  %i\n", input_blob, inputIndex);
	
#if NV_TENSORRT_MAJOR > 1
	nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));

#if NV_TENSORRT_MAJOR >= 7
    if( mModelType == MODEL_ONNX )
        inputDims = shiftDims(inputDims);   // change NCHW to CHW if EXPLICIT_BATCH set
#endif
#else
    Dims3 inputDims = engine->getBindingDimensions(inputIndex);
#endif

	size_t inputSize = maxBatchSize * DIMS_C(inputDims) * DIMS_H(inputDims) * DIMS_W(inputDims) * sizeof(float);
	printf(LOG_TRT "binding to input 0 %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", input_blob, maxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);
	

	/*
	 * allocate memory to hold the input buffer
	 */
	if( !cudaAllocMapped((void**)&mInputCPU, (void**)&mInputCUDA, inputSize) )
	{
		printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
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
		printf(LOG_TRT "binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

	#if NV_TENSORRT_MAJOR > 1
		nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));

    #if NV_TENSORRT_MAJOR >= 7
        if( mModelType == MODEL_ONNX )
            outputDims = shiftDims(outputDims);  // change NCHW to CHW if EXPLICIT_BATCH set
    #endif
	#else
		Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	#endif

		size_t outputSize = maxBatchSize * DIMS_C(outputDims) * DIMS_H(outputDims) * DIMS_W(outputDims) * sizeof(float);
		printf(LOG_TRT "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), maxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		//if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			printf(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
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
	

	/*
	 * create events for timing
	 */
	for( int n=0; n < PROFILER_TOTAL * 2; n++ )
		CUDA(cudaEventCreate(&mEventsGPU[n]));


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
}	

