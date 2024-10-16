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

#if NV_TENSORRT_MAJOR < 10
#include "NvCaffeParser.h"
#include "NvUffParser.h"
#endif

#if NV_TENSORRT_MAJOR >= 5
#include "NvOnnxParser.h"
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

#if NV_TENSORRT_MAJOR < 10
    #define TRT_DESTROY(x) x->destroy()
#else
    #define TRT_DESTROY(x) delete x
#endif

#define LOG_DOWNLOADER_TOOL "        if loading a built-in model, maybe it wasn't downloaded before.\n\n"    \
					   "        Run the Model Downloader tool again and select it for download:\n\n"   \
					   "           $ cd <jetson-inference>/tools\n" 	  	\
					   "           $ ./download-models.sh\n"

#define USE_INPUT_TENSOR_CUDA_DEVICE_MEMORY
#define CHECKSUM_TYPE "sha256sum"

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
    return nullptr;
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

#if NV_TENSORRT_MAJOR >= 8
static inline bool isFp16Enabled( nvinfer1::IBuilderConfig* config )
{
	return config->getFlag(nvinfer1::BuilderFlag::kFP16);
}

static inline bool isInt8Enabled( nvinfer1::IBuilderConfig* config )
{
	return config->getFlag(nvinfer1::BuilderFlag::kINT8);
}
#else // NV_TENSORRT_MAJOR <= 7
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
#endif

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

	LogWarning(LOG_TRT "warning -- unknown nvinfer1::DataType (%i)\n", (int)type);
	return "UNKNOWN";
}

#if NV_TENSORRT_MAJOR <= 7
static inline const char* dimensionTypeToStr( nvinfer1::DimensionType type )
{
	switch(type)
	{
		case nvinfer1::DimensionType::kSPATIAL:	 return "SPATIAL";
		case nvinfer1::DimensionType::kCHANNEL:	 return "CHANNEL";
		case nvinfer1::DimensionType::kINDEX:	 return "INDEX";
		case nvinfer1::DimensionType::kSEQUENCE: return "SEQUENCE";
	}

	LogWarning(LOG_TRT "warning -- unknown nvinfer1::DimensionType (%i)\n", (int)type);
	return "UNKNOWN";
}
#endif
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

static inline void copyDims( Dims3* dest, const nvinfer1::Dims* src )
{
	for( int n=0; n < src->nbDims; n++ )
		dest->d[n] = src->d[n];
	
	dest->nbDims = src->nbDims;
}

static inline size_t sizeDims( const nvinfer1::Dims& dims, const size_t elementSize=1 )
{
	size_t sz = dims.d[0];
	
	for ( int n=1; n < dims.nbDims; n++ )
		sz *= dims.d[n];

	return sz * elementSize;
}
#else
static inline nvinfer1::Dims3 validateDims( const nvinfer1::Dims3& dims )
{
	nvinfer1::Dims3 out = dims;
	
	if( DIMS_C(out) == 0 )
		DIMS_C(out) = 1;
	
	if( DIMS_H(out) == 0 )
		DIMS_H(out) = 1;
	
	if( DIMS_W(out) == 0 )
		DIMS_W(out) = 1;
	
	return out;
}

static inline void copyDims( Dims3* dest, const nvinfer1::Dims3* src )
{
	memcpy(dest, src, sizeof(nvinfer1::Dims));
}

static inline size_t sizeDims( const Dims3& dims, const size_t elementSize=1 )
{
	return DIMS_C(dims) * DIMS_H(dims) * DIMS_W(dims) * elementSize;
}
#endif	

#if NV_TENSORRT_MAJOR >= 7
static inline nvinfer1::Dims shiftDims( const nvinfer1::Dims& dims )
{
	// TensorRT 7.0 requires EXPLICIT_BATCH flag for ONNX models,
	// which adds a batch dimension (4D NCHW), whereas historically
	// 3D CHW was expected.  Remove the batch dim (it is typically 1)
	nvinfer1::Dims out = dims;
	
	/*out.d[0] = dims.d[1];
	out.d[1] = dims.d[2];
	out.d[2] = dims.d[3];
	out.d[3] = 1;*/
	
	if( dims.nbDims == 1 )
		return out;
	
	for( int n=0; n < dims.nbDims; n++ )
		out.d[n] = dims.d[n+1];
	
	for( int n=dims.nbDims; n < nvinfer1::Dims::MAX_DIMS; n++ )
		out.d[n] = 1;
	
	out.nbDims -= 1;
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
    return nullptr;
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
        default:            return nvinfer1::DeviceType::kGPU;
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
		case MODEL_ENGINE:	return "engine";
	}
    return nullptr;
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
	else if( strcasecmp(str, "engine") == 0 || strcasecmp(str, "plan") == 0 || strcasecmp(str, "trt") == 0 )
		return MODEL_ENGINE;

	return MODEL_CUSTOM;
}

modelType modelTypeFromPath( const char* path )
{
	if( !path )
		return MODEL_CUSTOM;
	
	return modelTypeFromStr(fileExtension(path).c_str());
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
    return nullptr;
}

//---------------------------------------------------------------------
tensorNet::Logger tensorNet::gLogger;

// constructor
tensorNet::tensorNet()
{
	mEngine   = NULL;
	mInfer    = NULL;
	mContext  = NULL;
	mStream   = NULL;
	mBindings	= NULL;

	mMaxBatchSize   = 0;	
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

#if NV_TENSORRT_MAJOR > 5
	mWorkspaceSize = 32 << 20;
#else
	mWorkspaceSize = 16 << 20;
#endif
}


// Destructor
tensorNet::~tensorNet()
{
	if( mContext != NULL )
	{
		TRT_DESTROY(mContext);
		mContext = NULL;
	}
	
	if( mEngine != NULL )
	{
		TRT_DESTROY(mEngine);
		mEngine = NULL;
	}
		
	if( mInfer != NULL )
	{
		TRT_DESTROY(mInfer);
		mInfer = NULL;
	}
	
	for( size_t n=0; n < mInputs.size(); n++ )
	{
	#ifdef USE_INPUT_TENSOR_CUDA_DEVICE_MEMORY
		CUDA_FREE(mInputs[n].CUDA);
	#else
		CUDA_FREE_HOST(mInputs[n].CUDA);
	#endif
	}
	
	for( size_t n=0; n < mOutputs.size(); n++ )
		CUDA_FREE_HOST(mOutputs[n].CPU);
	
	free(mBindings);
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
		LogError(LOG_TRT "DetectNativePrecisions() failed to create TensorRT IBuilder instance\n");
		return types;
	}

#if NV_TENSORRT_MAJOR >= 5 && NV_TENSORRT_MAJOR <= 7
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

	LogVerbose(LOG_TRT "native precisions detected for %s:  ", deviceTypeToStr(device));
 
	for( uint32_t n=0; n < numTypes; n++ )
	{
		LogVerbose("%s", precisionTypeToStr(types[n]));

		if( n < numTypes - 1 )
			LogVerbose(", ");
	}

	LogVerbose("\n");
	TRT_DESTROY(builder);
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


// SelectPrecision
precisionType tensorNet::SelectPrecision( precisionType precision, deviceType device, bool allowInt8 )
{
	LogVerbose(LOG_TRT "desired precision specified for %s: %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));

	if( precision == TYPE_DISABLED )
	{
		LogWarning(LOG_TRT "skipping network specified with precision TYPE_DISABLE\n");
		LogWarning(LOG_TRT "please specify a valid precision to create the network\n");
	}
	else if( precision == TYPE_FASTEST )
	{
		if( !allowInt8 )
			LogWarning(LOG_TRT "requested fasted precision for device %s without providing valid calibrator, disabling INT8\n", deviceTypeToStr(device));

		precision = FindFastestPrecision(device, allowInt8);
		LogVerbose(LOG_TRT "selecting fastest native precision for %s:  %s\n", deviceTypeToStr(device), precisionTypeToStr(precision));
	}
	else
	{
		if( !DetectNativePrecision(precision, device) )
		{
			LogWarning(LOG_TRT "precision %s is not supported for device %s\n", precisionTypeToStr(precision), deviceTypeToStr(device));
			precision = FindFastestPrecision(device, allowInt8);
			LogWarning(LOG_TRT "falling back to fastest precision for device %s (%s)\n", deviceTypeToStr(device), precisionTypeToStr(precision));
		}

		if( precision == TYPE_INT8 && !allowInt8 )
			LogWarning(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
	}

	return precision;
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
					    const std::vector<std::string>& inputs, 
					    const std::vector<Dims3>& inputDims,
					    const std::vector<std::string>& outputs,    // network outputs
					    unsigned int maxBatchSize,			   // batch size - NB must be at least as large as the batch we want to run with
					    precisionType precision, 
					    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, 	
					    char** engineStream, size_t* engineSize)	   // output stream for the GIE model
{
	if( !engineStream || !engineSize )
		return false;

	// create builder and network definition interfaces
	nvinfer1::IBuilder* builder = CREATE_INFER_BUILDER(gLogger);
	
#if NV_TENSORRT_MAJOR >= 8
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(0);
#else
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
#endif

	LogInfo(LOG_TRT "device %s, loading %s %s\n", deviceTypeToStr(device), deployFile.c_str(), modelFile.c_str());
	
	// parse the different types of model formats
	if( mModelType == MODEL_CAFFE )
	{
	#if NV_TENSORRT_MAJOR < 10
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
			LogError(LOG_TRT "device %s, failed to parse caffe network\n", deviceTypeToStr(device));
			return false;
		}

		// the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate	
		const size_t num_outputs = outputs.size();
		
		for( size_t n=0; n < num_outputs; n++ )
		{
			nvinfer1::ITensor* tensor = blobNameToTensor->find(outputs[n].c_str());
		
			if( !tensor )
			{
				LogError(LOG_TRT "failed to retrieve tensor for Output \"%s\"\n", outputs[n].c_str());
			}
			else
			{
			#if NV_TENSORRT_MAJOR >= 4
				nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(tensor->getDimensions());
				LogVerbose(LOG_TRT "retrieved Output tensor \"%s\":  %ix%ix%i\n", tensor->getName(), dims.d[0], dims.d[1], dims.d[2]);
			#endif
			}

			network->markOutput(*tensor);
		}

		//parser->destroy();
    #else
        LogError(LOG_TRT "TensorRT %i.%i does not support legacy caffe models\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
        return false;
    #endif
	}
#if NV_TENSORRT_MAJOR >= 5
	else if( mModelType == MODEL_ONNX )
	{
	#if NV_TENSORRT_MAJOR >= 7
		TRT_DESTROY(network);
		network = builder->createNetworkV2(1U << (uint32_t)nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

		if( !network )
		{
		  LogError(LOG_TRT "IBuilder::createNetworkV2(EXPLICIT_BATCH) failed\n");
		  return false;
		}
	#endif

		nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

		if( !parser )
		{
			LogError(LOG_TRT "failed to create nvonnxparser::IParser instance\n");
			return false;
		}

    #if NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kINFO;
    #else
        const int parserLogLevel = (int)nvinfer1::ILogger::Severity::kVERBOSE;
    #endif

		if( !parser->parseFromFile(modelFile.c_str(), parserLogLevel) )
		{
			LogError(LOG_TRT "failed to parse ONNX model '%s'\n", modelFile.c_str());
			return false;
		}

		//parser->destroy();
	}
	else if( mModelType == MODEL_UFF )
	{
	#if NV_TENSORRT_MAJOR < 10
		// create parser instance
		nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
		
		if( !parser )
		{
			LogError(LOG_TRT "failed to create UFF parser\n");
			return false;
		}
		
		// register inputs
		const size_t numInputs = inputs.size();

		for( size_t n=0; n < numInputs; n++ )
		{
			if( !parser->registerInput(inputs[n].c_str(), inputDims[n], nvuffparser::UffInputOrder::kNCHW) )
			{
				LogError(LOG_TRT "failed to register input '%s' for UFF model '%s'\n", inputs[n].c_str(), modelFile.c_str());
				return false;
			}
		}

		// register outputs
		/*const size_t numOutputs = outputs.size();
		
		for( uint32_t n=0; n < numOutputs; n++ )
		{
			if( !parser->registerOutput(outputs[n].c_str()) )
				printf(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", outputs[n].c_str(), modelFile.c_str());
		}*/

		// UFF outputs are forwarded to 'MarkOutput_0'
		if( !parser->registerOutput("MarkOutput_0") )
			LogError(LOG_TRT "failed to register output '%s' for UFF model '%s'\n", "MarkOutput_0", modelFile.c_str());

		// parse network
		if( !parser->parse(modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT) )
		{
			LogError(LOG_TRT "failed to parse UFF model '%s'\n", modelFile.c_str());
			return false;
		}
		
		//parser->destroy();
    #else
        LogError(LOG_TRT "TensorRT %i.%i does not support legacy caffe models\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
        return false;
    #endif
	}
#endif

#if NV_TENSORRT_MAJOR >= 4
	if( precision == TYPE_INT8 && !calibrator )
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
			LogVerbose(LOG_TRT "retrieved Input tensor '%s':  %ix%ix%i\n", network->getInput(i)->getName(), dims.d[0], dims.d[1], dims.d[2]);
		}

		// default to random calibration
		calibrator = new randInt8Calibrator(1, mCacheCalibrationPath, inputDimensions);
		LogWarning(LOG_TRT "warning:  device %s using INT8 precision with RANDOM calibration\n", deviceTypeToStr(device));
	}
#endif

	// configure the builder
#if NV_TENSORRT_MAJOR >= 8
	nvinfer1::IBuilderConfig* builderConfig = builder->createBuilderConfig();
	
	if( !ConfigureBuilder(builder, builderConfig, maxBatchSize, mWorkspaceSize, 
					  precision, device, allowGPUFallback, calibrator) )
	{
		LogError(LOG_TRT "device %s, failed to configure builder\n", deviceTypeToStr(device));
		return false;
	}
	
	// attempt to load the timing cache
	const nvinfer1::ITimingCache* timingCache = NULL;
	
	char timingCachePath[PATH_MAX];
	sprintf(timingCachePath, "/usr/local/bin/networks/tensorrt.%i.timingcache", NV_TENSORRT_VERSION);

	if( fileExists(timingCachePath) )
	{
		LogInfo(LOG_TRT "loading timing cache from %s\n", timingCachePath);
		
		void* timingCacheBuffer = NULL;
		const size_t timingCacheSize = loadFile(timingCachePath, &timingCacheBuffer);
		
		if( timingCacheSize > 0 )
		{
			timingCache = builderConfig->createTimingCache(timingCacheBuffer, timingCacheSize);
			free(timingCacheBuffer);
		}
	}
	
	if( !timingCache )
	{
		timingCache = builderConfig->createTimingCache(NULL, 0);  // create a new cache
	
		if( !timingCache )
			LogWarning(LOG_TRT "couldn't create new timing cache\n");
	}
	
	if( timingCache != NULL && !builderConfig->setTimingCache(*timingCache, false) )
		LogWarning(LOG_TRT "failed to activate timing cache");
#else
	if( !ConfigureBuilder(builder, maxBatchSize, mWorkspaceSize, precision, device, allowGPUFallback, calibrator) )
	{
		LogError(LOG_TRT "device %s, failed to configure builder\n", deviceTypeToStr(device));
		return false;
	}
#endif

	// build CUDA engine
	LogInfo(LOG_TRT "device %s, building CUDA engine (this may take a few minutes the first time a network is loaded)\n", deviceTypeToStr(device));

	if( Log::GetLevel() < Log::VERBOSE )
		LogInfo(LOG_TRT "info: to see status updates during engine building, enable verbose logging with --verbose\n");

#if NV_TENSORRT_MAJOR >= 10
    nvinfer1::IHostMemory* serMem = builder->buildSerializedNetwork(*network, *builderConfig);
#else
#if NV_TENSORRT_MAJOR >= 8
	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *builderConfig);
#else
	nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
#endif

	if( !engine )
	{
		LogError(LOG_TRT "device %s, failed to build CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	LogSuccess(LOG_TRT "device %s, completed building CUDA engine\n", deviceTypeToStr(device));
#endif

#if NV_TENSORRT_MAJOR >= 8
	if( timingCache != NULL )
	{
		// save the updated timing cache
		nvinfer1::IHostMemory* timingCacheMem = timingCache->serialize();
		
		if( timingCacheMem != NULL )
		{
			const char* timingCacheBuffer = (char*)timingCacheMem->data();
			const size_t timingCacheSize = timingCacheMem->size();
		
			LogVerbose(LOG_TRT "saving timing cache to %s (%zu bytes)\n", timingCachePath, timingCacheSize);
			
			// write the cache file
			FILE* timingCacheFile = NULL;
			timingCacheFile = fopen(timingCachePath, "wb");

			if( timingCacheFile != NULL )
			{
				if( fwrite(timingCacheBuffer,	1, timingCacheSize, timingCacheFile) != timingCacheSize )
					LogWarning(LOG_TRT "failed to write %zu bytes to timing cache file %s\n", timingCacheSize, timingCachePath);
			
				fclose(timingCacheFile);
			}
			else
			{
				LogWarning(LOG_TRT "failed to open timing cache file for writing %s\n", timingCachePath);
			}
			
			TRT_DESTROY(timingCacheMem);
		}
	
		delete timingCache;
	}
	
	TRT_DESTROY(builderConfig);
#endif
	
	// we don't need the network definition any more, and we can destroy the parser
	TRT_DESTROY(network);
	//parser->destroy();
	
#if NV_TENSORRT_MAJOR >= 2
#if NV_TENSORRT_MAJOR < 10
	nvinfer1::IHostMemory* serMem = engine->serialize();  // serialize the engine
#endif

	if( !serMem )
	{
		LogError(LOG_TRT "device %s, failed to serialize CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	const char* serData = (char*)serMem->data();
	const size_t serSize = serMem->size();

	// allocate memory to store the bitstream
	char* engineMemory = (char*)malloc(serSize);

	if( !engineMemory )
	{
		LogError(LOG_TRT "failed to allocate %zu bytes to store CUDA engine\n", serSize);
		return false;
	}

	memcpy(engineMemory, serData, serSize);
	
	*engineStream = engineMemory;
	*engineSize = serSize;
	
	TRT_DESTROY(serMem);
#else
	engine->serialize(modelStream);
#endif

#if NV_TENSORRT_MAJOR < 10
	TRT_DESTROY(engine);
#endif

	TRT_DESTROY(builder);
	
	return true;
}


// ConfigureBuilder
#if NV_TENSORRT_MAJOR >= 8
bool tensorNet::ConfigureBuilder( nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config,  
				    		    uint32_t maxBatchSize, uint32_t workspaceSize, precisionType precision, 
				    		    deviceType device, bool allowGPUFallback, 
				    		    nvinfer1::IInt8Calibrator* calibrator )
{
	if( !builder )
		return false;

	LogVerbose(LOG_TRT "device %s, configuring network builder\n", deviceTypeToStr(device));
		
#if NV_TENSORRT_MAJOR < 10
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(workspaceSize); // config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE) in TRT10 - but now already defaults to the max
	config->setMinTimingIterations(3);          // allow time for GPU to spin up
#endif

	config->setAvgTimingIterations(2);

	if( mEnableDebug )
		config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
	
	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		//config->setFlag(nvinfer1::BuilderFlag::kFP16); // TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
			LogError(LOG_TRT "device %s, INT8 requested but calibrator is NULL\n", deviceTypeToStr(device));
			return false;
		}

		config->setInt8Calibrator(calibrator);
	}
	else if( precision == TYPE_FP16 )
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}
	
	// set the default device type
	config->setDefaultDeviceType(deviceTypeToTRT(device));

	if( allowGPUFallback )
		config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);

	LogInfo(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), isFp16Enabled(config) ? "ON" : "OFF"); 
	LogInfo(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), isInt8Enabled(config) ? "ON" : "OFF"); 
	LogInfo(LOG_TRT "device %s, workspace size: %u\n", deviceTypeToStr(device), workspaceSize);

	return true;
}

#else  // NV_TENSORRT_MAJOR <= 7
	
bool tensorNet::ConfigureBuilder( nvinfer1::IBuilder* builder, uint32_t maxBatchSize, 
				    		    uint32_t workspaceSize, precisionType precision, 
				    		    deviceType device, bool allowGPUFallback, 
				    		    nvinfer1::IInt8Calibrator* calibrator )
{
	if( !builder )
		return false;

	LogVerbose(LOG_TRT "device %s, configuring network builder\n", deviceTypeToStr(device));
		
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(workspaceSize);

	builder->setDebugSync(mEnableDebug);
	builder->setMinFindIterations(3);	// allow time for GPU to spin up
	builder->setAverageFindIterations(2);

	// set up the builder for the desired precision
	if( precision == TYPE_INT8 )
	{
	#if NV_TENSORRT_MAJOR >= 4
		builder->setInt8Mode(true);
		//builder->setFp16Mode(true);		// TODO:  experiment for benefits of both INT8/FP16
		
		if( !calibrator )
		{
			LogError(LOG_TRT "device %s, INT8 requested but calibrator is NULL\n", deviceTypeToStr(device));
			return false;
		}

		builder->setInt8Calibrator(calibrator);
	#else
		LogError(LOG_TRT "INT8 precision requested, and TensorRT %u.%u doesn't meet minimum version for INT8\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		LogError(LOG_TRT "please use minumum version of TensorRT 4.0 or newer for INT8 support\n");

		return false;
	#endif
	}
	else if( precision == TYPE_FP16 )
	{
	#if NV_TENSORRT_MAJOR >= 4
		builder->setFp16Mode(true);
	#else
		builder->setHalf2Mode(true);
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
		LogError(LOG_TRT "device %s is not supported in TensorRT %u.%u\n", deviceTypeToStr(device), NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif

	LogInfo(LOG_TRT "device %s, building FP16:  %s\n", deviceTypeToStr(device), isFp16Enabled(builder) ? "ON" : "OFF"); 
	LogInfo(LOG_TRT "device %s, building INT8:  %s\n", deviceTypeToStr(device), isInt8Enabled(builder) ? "ON" : "OFF"); 
	LogInfo(LOG_TRT "device %s, workspace size: %u\n", deviceTypeToStr(device), workspaceSize);

	return true;
}
#endif


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
	return LoadNetwork(prototxt_path, model_path, mean_path,
				    input_blob, Dims3(1,1,1), output_blobs,
				    maxBatchSize, precision, device,
				    allowGPUFallback, calibrator, stream);
}


// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, 
					    const std::vector<std::string>& input_blobs, 
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	std::vector<Dims3> input_dims;

	for( size_t n=0; n < input_blobs.size(); n++ )
		input_dims.push_back(Dims3(1,1,1));

	return LoadNetwork(prototxt_path, model_path, mean_path,
				    input_blobs, input_dims, output_blobs,
				    maxBatchSize, precision, device,
				    allowGPUFallback, calibrator, stream);
}


// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path, const char* model_path, const char* mean_path, 
					    const char* input_blob, const Dims3& input_dim,
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
	std::vector<std::string> inputs;
	std::vector<Dims3> input_dims;

	inputs.push_back(input_blob);
	input_dims.push_back(input_dim);

	return LoadNetwork(prototxt_path, model_path, mean_path,
				    inputs, input_dims, output_blobs,
				    maxBatchSize, precision, device,
				    allowGPUFallback, calibrator, stream);
}

		   
// LoadNetwork
bool tensorNet::LoadNetwork( const char* prototxt_path_, const char* model_path_, const char* mean_path, 
					    const std::vector<std::string>& input_blobs, 
					    const std::vector<Dims3>& input_dims,
					    const std::vector<std::string>& output_blobs, 
					    uint32_t maxBatchSize, precisionType precision,
				   	    deviceType device, bool allowGPUFallback,
					    nvinfer1::IInt8Calibrator* calibrator, cudaStream_t stream )
{
#if NV_TENSORRT_MAJOR >= 4
	LogInfo(LOG_TRT "TensorRT version %u.%u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
#else
	LogInfo(LOG_TRT "TensorRT version %u.%u\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
#endif

	/*
	 * validate arguments
	 */
	if( !model_path_ )
	{
		LogError(LOG_TRT "model path was NULL - must have valid model path to LoadNetwork()\n");
		return false;
	}

	if( input_blobs.size() == 0 || output_blobs.size() == 0 )
	{
		LogError(LOG_TRT "requested number of input layers or output layers was zero\n");
		return false;
	}

	if( input_blobs.size() != input_dims.size() )
	{
		LogError(LOG_TRT "input mismatch - requested %zu input layers, but only %zu input dims\n", input_blobs.size(), input_dims.size());
		return false;
	}

	
	/*
	 * load NV inference plugins
	 */
#if NV_TENSORRT_MAJOR > 4
	static bool loadedPlugins = false;

	if( !loadedPlugins )
	{
		LogVerbose(LOG_TRT "loading NVIDIA plugins...\n");

		loadedPlugins = initLibNvInferPlugins(&gLogger, "");

		if( !loadedPlugins )
		{
			LogError(LOG_TRT "failed to load NVIDIA plugins\n");
		}
		else
		{
			LogVerbose(LOG_TRT "completed loading NVIDIA plugins.\n");
		}
	}
#endif

	/*
	 * verify the prototxt and model paths
	 */
	const std::string model_path    = locateFile(model_path_);
	const std::string prototxt_path = locateFile(prototxt_path_ != NULL ? prototxt_path_ : "");
	
	const std::string model_ext = fileExtension(model_path_);
	const modelType   model_fmt = modelTypeFromStr(model_ext.c_str());

	LogVerbose(LOG_TRT "detected model format - %s  (extension '.%s')\n", modelTypeToStr(model_fmt), model_ext.c_str());

	if( model_fmt == MODEL_CUSTOM )
	{
		LogError(LOG_TRT "model format '%s' not supported by jetson-inference\n", modelTypeToStr(model_fmt));
		return false;
	}
#if NV_TENSORRT_MAJOR < 5
	else if( model_fmt == MODEL_ONNX )
	{
		LogError(LOG_TRT "importing ONNX models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
	else if( model_fmt == MODEL_UFF )
	{
		LogError(LOG_TRT "importing UFF models is not supported in TensorRT %u.%u (version >= 5.0 required)\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR);
		return false;
	}
#endif
	else if( model_fmt == MODEL_CAFFE && !prototxt_path_ )
	{
		LogError(LOG_TRT "attempted to load caffe model without specifying prototxt file\n");
		return false;
	}
	else if( model_fmt == MODEL_ENGINE )
	{
		if( !LoadEngine(model_path.c_str(), input_blobs, output_blobs, NULL, device, stream) )
		{
			LogError(LOG_TRT "failed to load %s\n", model_path.c_str());
			return false;
		}

		mModelType = model_fmt;
		mModelPath = model_path;
		mModelFile = pathFilename(mModelPath);
		
		LogSuccess(LOG_TRT "device %s, initialized %s\n", deviceTypeToStr(device), mModelPath.c_str());	
		return true;
	}

	mModelType = model_fmt;


	/*
	 * resolve the desired precision to a specific one that's available
	 */
	precision = SelectPrecision(precision, device, (calibrator != NULL));

	if( precision == TYPE_DISABLED )
		return false;

	
	/*
	 * attempt to load network engine from cache before profiling with tensorRT
	 */	
	char* engineStream = NULL;
	size_t engineSize = 0;

	char cache_prefix[PATH_MAX];
	char cache_path[PATH_MAX];

	sprintf(cache_prefix, "%s.%u.%u.%i.%s.%s", model_path.c_str(), maxBatchSize, (uint32_t)allowGPUFallback, NV_TENSORRT_VERSION, deviceTypeToStr(device), precisionTypeToStr(precision));
	sprintf(cache_path, "%s.calibration", cache_prefix);
	mCacheCalibrationPath = cache_path;
	
	sprintf(cache_path, "%s.%s", model_path.c_str(), CHECKSUM_TYPE);
	mChecksumPath = cache_path;
	
	sprintf(cache_path, "%s.engine", cache_prefix);
	mCacheEnginePath = cache_path;	

	// check for existence of cache
	if( !ValidateEngine(model_path.c_str(), cache_path, mChecksumPath.c_str()) )
	{
		LogVerbose(LOG_TRT "cache file invalid, profiling network model on device %s\n", deviceTypeToStr(device));
	
		// check for existence of model
		if( model_path.size() == 0 )
		{
			LogError("\nerror:  model file '%s' was not found.\n", model_path_);
			LogInfo("%s\n", LOG_DOWNLOADER_TOOL);
			return 0;
		}

		// parse the model and profile the engine
		if( !ProfileModel(prototxt_path, model_path, input_blobs, input_dims,
					   output_blobs, maxBatchSize, precision, device, 
					   allowGPUFallback, calibrator, &engineStream, &engineSize) )
		{
			LogError(LOG_TRT "device %s, failed to load %s\n", deviceTypeToStr(device), model_path_);
			return 0;
		}
	
		LogVerbose(LOG_TRT "network profiling complete, saving engine cache to %s\n", cache_path);
		
		// write the cache file
		FILE* cacheFile = NULL;
		cacheFile = fopen(cache_path, "wb");

		if( cacheFile != NULL )
		{
			if( fwrite(engineStream,	1, engineSize, cacheFile) != engineSize )
				LogError(LOG_TRT "failed to write %zu bytes to engine cache file %s\n", engineSize, cache_path);
		
			fclose(cacheFile);
		}
		else
		{
			LogError(LOG_TRT "failed to open engine cache file for writing %s\n", cache_path);
		}

		LogSuccess(LOG_TRT "device %s, completed saving engine cache to %s\n", deviceTypeToStr(device), cache_path);
		
		// write the checksum file
		LogVerbose(LOG_TRT "saving model checksum to %s\n", mChecksumPath.c_str());
		
		char cmd[PATH_MAX * 2 + 256];
		snprintf(cmd, sizeof(cmd), "%s %s | awk '{print $1}' > %s", CHECKSUM_TYPE, model_path.c_str(), mChecksumPath.c_str());
	
		LogVerbose(LOG_TRT "%s\n", cmd);
	
		const int result = system(cmd);
		
		if( result != 0 )
			LogError(LOG_TRT "failed to save model checksum to %s\n", mChecksumPath.c_str());
	}
	else
	{
		if( !LoadEngine(cache_path, &engineStream, &engineSize) )
			return false;
	}

	LogSuccess(LOG_TRT "device %s, loaded %s\n", deviceTypeToStr(device), model_path.c_str());
	

	/*
	 * create the runtime engine instance
	 */
	if( !LoadEngine(engineStream, engineSize, input_blobs, output_blobs, NULL, device, stream) )
	{
		LogError(LOG_TRT "failed to create TensorRT engine for %s, device %s\n", model_path.c_str(), deviceTypeToStr(device));
		return false;
	}

	free(engineStream); // not used anymore

	mPrototxtPath     = prototxt_path;
	mModelPath        = model_path;
	mModelFile        = pathFilename(mModelPath);
	mPrecision        = precision;
	mAllowGPUFallback = allowGPUFallback;
	mMaxBatchSize 	   = maxBatchSize;

	if( mean_path != NULL )
		mMeanPath = mean_path;

	LogInfo(LOG_TRT "\n");
	LogSuccess(LOG_TRT "device %s, %s initialized.\n", deviceTypeToStr(device), mModelPath.c_str());	
	
	return true;
}


// LoadEngine
bool tensorNet::LoadEngine( char* engine_stream, size_t engine_size,
			  		   const std::vector<std::string>& input_blobs, 
			  		   const std::vector<std::string>& output_blobs,
			  		   nvinfer1::IPluginFactory* pluginFactory,
					   deviceType device, cudaStream_t stream )
{
	/*
	 * create runtime inference engine execution context
	 */
	nvinfer1::IRuntime* infer = CREATE_INFER_RUNTIME(gLogger);
	
	if( !infer )
	{
		LogError(LOG_TRT "device %s, failed to create TensorRT runtime\n", deviceTypeToStr(device));
		return false;
	}

#if NV_TENSORRT_MAJOR >= 5 
#if !(NV_TENSORRT_MAJOR == 5 && NV_TENSORRT_MINOR == 0 && NV_TENSORRT_PATCH == 0)
	// if using DLA, set the desired core before deserialization occurs
	if( device == DEVICE_DLA_0 )
	{
		LogVerbose(LOG_TRT "device %s, enabling DLA core 0\n", deviceTypeToStr(device));
		infer->setDLACore(0);
	}
	else if( device == DEVICE_DLA_1 )
	{
		LogVerbose(LOG_TRT "device %s, enabling DLA core 1\n", deviceTypeToStr(device));
		infer->setDLACore(1);
	}
#endif
#endif

#if NV_TENSORRT_MAJOR >= 10
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(engine_stream, engine_size);
#elif NV_TENSORRT_MAJOR > 1
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(engine_stream, engine_size, pluginFactory);
#else
	nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(engine_stream, engine_size); //infer->deserializeCudaEngine(modelStream);
#endif

	if( !engine )
	{
		LogError(LOG_TRT "device %s, failed to create CUDA engine\n", deviceTypeToStr(device));
		return false;
	}

	if( !LoadEngine(engine, input_blobs, output_blobs, device, stream) )
	{
		LogError(LOG_TRT "device %s, failed to create resources for CUDA engine\n", deviceTypeToStr(device));
		return false;
	}	

	mInfer = infer;
	return true;
}


#if NV_TENSORRT_MAJOR >= 10
static int trtTensorIndex( nvinfer1::ICudaEngine* engine, const char* name )
{
    const int numBindings = engine->getNbIOTensors();
	
	for( int n=0; n < numBindings; n++ )
	{
		const char* bind_name = engine->getIOTensorName(n);

        if( strcmp(name, bind_name) == 0 )
            return n;
    }
    
    return -1;
}         
#endif


// LoadEngine
bool tensorNet::LoadEngine( nvinfer1::ICudaEngine* engine,
 			  		   const std::vector<std::string>& input_blobs, 
			  		   const std::vector<std::string>& output_blobs,
			  		   deviceType device, cudaStream_t stream)
{
	if( !engine )
		return NULL;

	nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	
	if( !context )
	{
		LogError(LOG_TRT "device %s, failed to create execution context\n", deviceTypeToStr(device));
		return 0;
	}

	if( mEnableDebug )
	{
		LogVerbose(LOG_TRT "device %s, enabling context debug sync.\n", deviceTypeToStr(device));
		context->setDebugSync(true);
	}

	if( mEnableProfiler )
		context->setProfiler(&gProfiler);

#if NV_TENSORRT_MAJOR < 10
	mMaxBatchSize = engine->getMaxBatchSize();
#else
    mMaxBatchSize = 1;
#endif

	LogInfo(LOG_TRT "\n");
	LogInfo(LOG_TRT "CUDA engine context initialized on device %s:\n", deviceTypeToStr(device));
	LogInfo(LOG_TRT "   -- layers       %i\n", engine->getNbLayers());
	LogInfo(LOG_TRT "   -- maxBatchSize %u\n", mMaxBatchSize);
	
#if NV_TENSORRT_MAJOR <= 7
	LogInfo(LOG_TRT "   -- workspace    %zu\n", engine->getWorkspaceSize());
#endif

#if NV_TENSORRT_MAJOR >= 10
    LogInfo(LOG_TRT "   -- deviceMemory %zu\n", engine->getDeviceMemorySize());
    LogInfo(LOG_TRT "   -- bindings     %i\n", engine->getNbIOTensors());
    
    const int numBindings = engine->getNbIOTensors();
	
	for( int n=0; n < numBindings; n++ )
	{
		LogInfo(LOG_TRT "   binding %i\n", n);

		const char* bind_name = engine->getIOTensorName(n);
        const bool bind_input = (engine->getTensorIOMode(bind_name) == nvinfer1::TensorIOMode::kINPUT);
        const bool bind_host = (engine->getTensorLocation(bind_name) == nvinfer1::TensorLocation::kHOST);
        
		LogInfo("                -- index   %i\n", n);
		LogInfo("                -- name    '%s'\n", bind_name);
		LogInfo("                -- type    %s\n", engine->getTensorFormatDesc(bind_name)); //dataTypeToStr(engine->getBindingDataType(n)));
		LogInfo("                -- in/out  %s\n", bind_input ? "INPUT" : "OUTPUT");
        LogInfo("                -- device  %s\n", bind_host ? "HOST" : "DEVICE");

		const nvinfer1::Dims bind_dims = engine->getTensorShape(bind_name);

		LogInfo("                -- # dims  %i\n", bind_dims.nbDims);
		
		for( int i=0; i < bind_dims.nbDims; i++ )
		#if NV_TENSORRT_MAJOR >= 8
			LogInfo("                -- dim #%i  %i\n", i, bind_dims.d[i]);	
		#else
			LogInfo("                -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));	
		#endif
	}

	LogInfo(LOG_TRT "\n");
#elif NV_TENSORRT_MAJOR >= 4
	LogInfo(LOG_TRT "   -- deviceMemory %zu\n", engine->getDeviceMemorySize());
	LogInfo(LOG_TRT "   -- bindings     %i\n", engine->getNbBindings());

	const int numBindings = engine->getNbBindings();
	
	for( int n=0; n < numBindings; n++ )
	{
		LogInfo(LOG_TRT "   binding %i\n", n);

		const char* bind_name = engine->getBindingName(n);

		LogInfo("                -- index   %i\n", n);
		LogInfo("                -- name    '%s'\n", bind_name);
		LogInfo("                -- type    %s\n", dataTypeToStr(engine->getBindingDataType(n)));
		LogInfo("                -- in/out  %s\n", engine->bindingIsInput(n) ? "INPUT" : "OUTPUT");
        
		const nvinfer1::Dims bind_dims = engine->getBindingDimensions(n);

		LogInfo("                -- # dims  %i\n", bind_dims.nbDims);
		
		for( int i=0; i < bind_dims.nbDims; i++ )
		#if NV_TENSORRT_MAJOR >= 8
			LogInfo("                -- dim #%i  %i\n", i, bind_dims.d[i]);	
		#else
			LogInfo("                -- dim #%i  %i (%s)\n", i, bind_dims.d[i], dimensionTypeToStr(bind_dims.type[i]));	
		#endif
	}

	LogInfo(LOG_TRT "\n");
#endif

	/*
	 * setup network input buffers
	 */
	const int numInputs = input_blobs.size();
	
	for( int n=0; n < numInputs; n++ )
	{
	#if NV_TENSORRT_MAJOR >= 10
	    const int inputIndex = trtTensorIndex(engine, input_blobs[n].c_str());
	#else
		const int inputIndex = engine->getBindingIndex(input_blobs[n].c_str());	
	#endif
		
		if( inputIndex < 0 )
		{
			LogError(LOG_TRT "failed to find requested input layer %s in network\n", input_blobs[n].c_str());
			return false;
		}

		LogVerbose(LOG_TRT "binding to input %i %s  binding index:  %i\n", n, input_blobs[n].c_str(), inputIndex);


        /*inputDims.nbDims = engine->getTensorComponentsPerElement(bind_name);
    
        for( int m=0; m < inputDims.nbDims; m++ )
            inputDims.d[m] = engine->getTensorBytesPerComponent(bind_name);*/
            
	#if NV_TENSORRT_MAJOR > 1
    #if NV_TENSORRT_MAJOR >= 10
        //mContext->setInputShape(input_blobs[n].c_str(), nvinfer1::Dims4{1, 3, 224, 224});
        nvinfer1::Dims inputDims = engine->getTensorShape(input_blobs[n].c_str());
    #else  
		nvinfer1::Dims inputDims = validateDims(engine->getBindingDimensions(inputIndex));
    #endif
	#if NV_TENSORRT_MAJOR >= 7
	    if( mModelType == MODEL_ONNX )
		   inputDims = shiftDims(inputDims);   // change NCHW to CHW if EXPLICIT_BATCH set
	#endif
	#else
		Dims3 inputDims = engine->getBindingDimensions(inputIndex);
	#endif

		const size_t inputSize = mMaxBatchSize * sizeDims(inputDims) * sizeof(float);
		LogVerbose(LOG_TRT "binding to input %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, input_blobs[n].c_str(), mMaxBatchSize, DIMS_C(inputDims), DIMS_H(inputDims), DIMS_W(inputDims), inputSize);

		// allocate memory to hold the input buffer
		void* inputCPU  = NULL;
		void* inputCUDA = NULL;

	#ifdef USE_INPUT_TENSOR_CUDA_DEVICE_MEMORY
		if( CUDA_FAILED(cudaMalloc((void**)&inputCUDA, inputSize)) )
		{
			LogError(LOG_TRT "failed to alloc CUDA device memory for tensor input, %zu bytes\n", inputSize);
			return false;
		}
		
		CUDA(cudaMemset(inputCUDA, 0, inputSize));
	#else
		if( !cudaAllocMapped((void**)&inputCPU, (void**)&inputCUDA, inputSize) )
		{
			LogError(LOG_TRT "failed to alloc CUDA mapped memory for tensor input, %zu bytes\n", inputSize);
			return false;
		}
	#endif
	
    #if 0 && NV_TENSORRT_MAJOR >= 10
        if( !mContext->setInputTensorAddress(input_blobs[n].c_str(), inputCUDA) )
        {
            LogError(LOG_TRT "failed to set input tensor address for %s (%zu bytes)\n", inputSize, input_blobs[n].c_str());
			return false;
        }
    #endif
	 
		layerInfo l;
		
		l.CPU  = (float*)inputCPU;
		l.CUDA = (float*)inputCUDA;
		l.size = inputSize;
		l.name = input_blobs[n];
		l.binding = inputIndex;
		
		copyDims(&l.dims, &inputDims);
		mInputs.push_back(l);
	}


	/*
	 * setup network output buffers
	 */
	const int numOutputs = output_blobs.size();
	
	for( int n=0; n < numOutputs; n++ )
	{
    #if NV_TENSORRT_MAJOR >= 10
	    const int outputIndex = trtTensorIndex(engine, output_blobs[n].c_str());
	#else
		const int outputIndex = engine->getBindingIndex(output_blobs[n].c_str());	
	#endif

		if( outputIndex < 0 )
		{
			LogError(LOG_TRT "failed to find requested output layer %s in network\n", output_blobs[n].c_str());
			return false;
		}

		LogVerbose(LOG_TRT "binding to output %i %s  binding index:  %i\n", n, output_blobs[n].c_str(), outputIndex);

    #if NV_TENSORRT_MAJOR >= 10
        nvinfer1::Dims outputDims = engine->getTensorShape(output_blobs[n].c_str());
	#elif NV_TENSORRT_MAJOR > 1
		nvinfer1::Dims outputDims = validateDims(engine->getBindingDimensions(outputIndex));

	#if NV_TENSORRT_MAJOR >= 7
		if( mModelType == MODEL_ONNX )
			outputDims = shiftDims(outputDims);  // change NCHW to CHW if EXPLICIT_BATCH set
	#endif
	#else
		Dims3 outputDims = engine->getBindingDimensions(outputIndex);
	#endif

		const size_t outputSize = mMaxBatchSize * sizeDims(outputDims) * sizeof(float);
		LogVerbose(LOG_TRT "binding to output %i %s  dims (b=%u c=%u h=%u w=%u) size=%zu\n", n, output_blobs[n].c_str(), mMaxBatchSize, DIMS_C(outputDims), DIMS_H(outputDims), DIMS_W(outputDims), outputSize);
	
		// allocate output memory 
		void* outputCPU  = NULL;
		void* outputCUDA = NULL;
		
		//if( CUDA_FAILED(cudaMalloc((void**)&outputCUDA, outputSize)) )
		if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputSize) )
		{
			LogError(LOG_TRT "failed to alloc CUDA mapped memory for tensor output, %zu bytes\n", outputSize);
			return false;
		}
	
    #if NV_TENSORRT_MAJOR >= 10
        if( !mContext->setTensorAddress(output_blobs[n].c_str(), outputCUDA) )
        {
            LogError(LOG_TRT "failed to set input tensor address for %s (%zu bytes)\n", outputSize, output_blobs[n].c_str());
			return false;
        }  
    #endif
    
		layerInfo l;
		
		l.CPU  = (float*)outputCPU;
		l.CUDA = (float*)outputCUDA;
		l.size = outputSize;
		l.name = output_blobs[n];
		l.binding = outputIndex;
		
		copyDims(&l.dims, &outputDims);
		mOutputs.push_back(l);
	}
	
	/*
	 * create list of binding buffers
	 */
	const int bindingSize = numBindings * sizeof(void*);

	mBindings = (void**)malloc(bindingSize);

	if( !mBindings )
	{
		LogError(LOG_TRT "failed to allocate %u bytes for bindings list\n", bindingSize);
		return false;
	}

	memset(mBindings, 0, bindingSize);

	for( uint32_t n=0; n < GetInputLayers(); n++ )
		mBindings[mInputs[n].binding] = mInputs[n].CUDA;

	for( uint32_t n=0; n < GetOutputLayers(); n++ )
		mBindings[mOutputs[n].binding] = mOutputs[n].CUDA;
	
	// find unassigned bindings and allocate them
	for( uint32_t n=0; n < numBindings; n++ )
	{
		if( mBindings[n] != NULL )
			continue;
		
    #if NV_TENSORRT_MAJOR >= 10
        const size_t bindingSize = sizeDims(validateDims(engine->getTensorShape(output_blobs[n].c_str()))) * mMaxBatchSize * sizeof(float);
    #else
		const size_t bindingSize = sizeDims(validateDims(engine->getBindingDimensions(n))) * mMaxBatchSize * sizeof(float);
    #endif
    
		if( CUDA_FAILED(cudaMalloc(&mBindings[n], bindingSize)) )
		{
			LogError(LOG_TRT "failed to allocate %zu bytes for unused binding %u\n", bindingSize, n);
			return false;
		}
		
		LogVerbose(LOG_TRT "allocated %zu bytes for unused binding %u\n", bindingSize, n);
	}
	

	/*
	 * create events for timing
	 */
	for( int n=0; n < PROFILER_TOTAL * 2; n++ )
		CUDA(cudaEventCreate(&mEventsGPU[n]));
	
	mEngine  = engine;
	mDevice  = device;
	mContext = context;
	
	SetStream(stream);	// set default device stream

	return true;
}


// LoadEngine
bool tensorNet::LoadEngine( const char* engine_filename,
			  		   const std::vector<std::string>& input_blobs, 
			  		   const std::vector<std::string>& output_blobs,
			  		   nvinfer1::IPluginFactory* pluginFactory,
					   deviceType device, cudaStream_t stream )
{
	char* engineStream = NULL;
	size_t engineSize = 0;

	// load the engine file contents
	if( !LoadEngine(engine_filename, &engineStream, &engineSize) )
		return false;

	// load engine resources from stream
	if( !LoadEngine(engineStream, engineSize, input_blobs, output_blobs,
				 pluginFactory, device, stream) )
	{
		free(engineStream);
		return false;
	}

	free(engineStream);
	return true;
}


// LoadEngine
bool tensorNet::LoadEngine( const char* filename, char** stream, size_t* size )
{
	if( !filename || !stream || !size )
		return false;

	LogInfo(LOG_TRT "loading network plan from engine cache... %s\n", filename);
		
	void* engineStream = NULL;
	const size_t engineSize = loadFile(filename, &engineStream);
	
	if( engineSize == 0 )
	{
		LogError(LOG_TRT "failed to load engine cache from %s\n", filename);
		return false;
	}
	
	*stream = (char*)engineStream;
	*size = engineSize;
	
	return true;
}


// ValidateEngine
bool tensorNet::ValidateEngine( const char* model_path, const char* cache_path, const char* checksum_path )
{
	// check for existence of cache
	if( !fileExists(cache_path) )
	{
		LogVerbose(LOG_TRT "could not find engine cache %s\n", cache_path);
		return false;
	}
	
	LogVerbose(LOG_TRT "found engine cache file %s\n", cache_path);
	
	// check for existence of checksum
	if( !fileExists(checksum_path) )
	{
		LogVerbose(LOG_TRT "could not find model checksum %s\n", checksum_path);
		return false;
	}
	
	LogVerbose(LOG_TRT "found model checksum %s\n", checksum_path);
	
	// validate that the checksum matches the original model
	char cmd[PATH_MAX * 2 + 256];
	snprintf(cmd, sizeof(cmd), "echo \"$(cat %s) %s\" | %s --check --status", checksum_path, model_path, CHECKSUM_TYPE);  // https://superuser.com/a/1468626
	
	LogVerbose(LOG_TRT "%s\n", cmd);
	
	const int result = system(cmd);
	
	if( result != 0 )
	{
		LogVerbose(LOG_TRT "model did not match checksum %s (return code %i)\n", checksum_path, result);
		return false;
	}
	
	LogVerbose(LOG_TRT "model matched checksum %s\n", checksum_path);
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


// ProcessNetwork
bool tensorNet::ProcessNetwork( bool sync )
{
	if( TENSORRT_VERSION_CHECK(8,4,1) && mModelType == MODEL_ONNX )
	{
	#if TENSORRT_VERSION_CHECK(8,4,1)
		// on TensorRT 8.4.1 (JetPack 5.0.2 / L4T R35.1.0) and newer, this warning appears:
		// the execute() method has been deprecated when used with engines built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. Please use executeV2() instead.
		// also, the batchSize argument passed into this function has no effect on changing the input shapes. Please use setBindingDimensions() function to change input shapes instead.
		if( sync )
		{
			if( !mContext->executeV2(mBindings) )
			{
				LogError(LOG_TRT "failed to execute TensorRT context on device %s\n", deviceTypeToStr(mDevice));
				return false;
			}
		}
		else
		{
		#if TENSORRT_VERSION_CHECK(10,0,0)
		    if( !mContext->enqueueV3(mStream) )
			{
				LogError(LOG_TRT "failed to enqueue TensorRT context on device %s\n", deviceTypeToStr(mDevice));
				return false;
			}
		#else
			if( !mContext->enqueueV2(mBindings, mStream, NULL) )
			{
				LogError(LOG_TRT "failed to enqueue TensorRT context on device %s\n", deviceTypeToStr(mDevice));
				return false;
			}
	    #endif
		}
	#endif
	}
	#if NV_TENSORRT_MAJOR < 10
	else
	{
		if( sync )
		{
			if( !mContext->execute(1, mBindings) )
			{
				LogError(LOG_TRT "failed to execute TensorRT context on device %s\n", deviceTypeToStr(mDevice));
				return false;
			}
		}
		else
		{
			if( !mContext->enqueue(1, mBindings, mStream, NULL) )
			{
				LogError(LOG_TRT "failed to enqueue TensorRT context on device %s\n", deviceTypeToStr(mDevice));
				return false;
			}
		}
	}
	#endif
	
	return true;
}


// validateClassLabels
static bool validateClassLabels( std::vector<std::string>& descriptions, std::vector<std::string>& synsets, int expectedClasses )
{
	const int numLoaded = descriptions.size();
	LogVerbose(LOG_TRT "loaded %i class labels\n", numLoaded);
	
	if( expectedClasses > 0 )
	{
		if( numLoaded != expectedClasses )
			LogError(LOG_TRT "didn't load expected number of class descriptions  (%i of %i)\n", numLoaded, expectedClasses);

		if( numLoaded < expectedClasses )
		{
			LogWarning(LOG_TRT "filling in remaining %i class descriptions with default labels\n", (expectedClasses - numLoaded));
	
			for( int n=numLoaded; n < expectedClasses; n++ )
			{
				char synset[10];
				sprintf(synset, "n%08i", n);

				char desc[64];
				sprintf(desc, "Class #%i", n);

				synsets.push_back(synset);
				descriptions.push_back(desc);
			}
		}
	}
	else if( numLoaded == 0 )
	{
		return false;
	}
	
	/*for( uint32_t n=0; n < descriptions.size(); n++ )
		LogVerbose(LOG_TRT "detectNet -- class label #%u:  '%s'\n", n, descriptions[n].c_str());*/
	
	return true;
}

	
// LoadClassLabels
bool tensorNet::LoadClassLabels( const char* filename, std::vector<std::string>& descriptions, std::vector<std::string>& synsets, int expectedClasses )
{
	if( !filename )
		return validateClassLabels(descriptions, synsets, expectedClasses);
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "tensorNet::LoadClassLabels() failed to find %s\n", filename);
		return validateClassLabels(descriptions, synsets, expectedClasses);
	}

	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "tensorNet::LoadClassLabels() failed to open %s\n", path.c_str());
		return validateClassLabels(descriptions, synsets, expectedClasses);
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
	return validateClassLabels(descriptions, synsets, expectedClasses);
}


// LoadClassLabels
bool tensorNet::LoadClassLabels( const char* filename, std::vector<std::string>& descriptions, int expectedClasses )
{
	std::vector<std::string> synsets;
	return LoadClassLabels(filename, descriptions, synsets, expectedClasses);
}


// validateClassColors
static bool validateClassColors( float4* colors, int numLoaded, int expectedClasses, float defaultAlpha )
{
	LogVerbose(LOG_TRT "loaded %i class colors\n", numLoaded);
	
	if( expectedClasses > 0 )
	{
		if( numLoaded != expectedClasses )
			LogWarning(LOG_TRT "didn't load expected number of class colors  (%i of %i)\n", numLoaded, expectedClasses);

		if( numLoaded < expectedClasses )
		{
			LogWarning(LOG_TRT "filling in remaining %i class colors with default colors\n", (expectedClasses - numLoaded));
	
			for( int n=numLoaded; n < expectedClasses; n++ )
			{
				colors[n] = tensorNet::GenerateColor(n, defaultAlpha);
				//LogVerbose(LOG_TRT "class color %i  (%f %f %f %f\n", n, colors[n].x, colors[n].y, colors[n].z, colors[n].w);
			}
		}
	}
	else if( numLoaded == 0 )
	{
		return false;
	}

	return true;
}


// LoadClassColors
bool tensorNet::LoadClassColors( const char* filename, float4* colors, int expectedClasses, float defaultAlpha )
{
	// validate parameters
	if( !colors || expectedClasses <= 0 )
	{
		LogError(LOG_TRT "tensorNet::LoadClassColors() had invalid/NULL parameters\n");
		return false;
	}
	
	if( !filename )
		return validateClassColors(colors, 0, expectedClasses, defaultAlpha);
	
	// locate the file
	const std::string path = locateFile(filename);

	if( path.length() == 0 )
	{
		LogError(LOG_TRT "tensorNet::LoadClassColors() failed to find %s\n", filename);
		return validateClassColors(colors, 0, expectedClasses, defaultAlpha);
	}
	
	// open the file
	FILE* f = fopen(path.c_str(), "r");
	
	if( !f )
	{
		LogError(LOG_TRT "tensorNet::LoadClassColors() failed to open %s\n", path.c_str());
		return validateClassColors(colors, 0, expectedClasses, defaultAlpha);
	}
	
	// read class colors
	char str[512];
	int numLoaded = 0;

	while( fgets(str, 512, f) != NULL && numLoaded < expectedClasses )
	{
		const int len = strlen(str);
		
		if( len <= 0 )
			continue;
		
		if( str[len-1] == '\n' )
			str[len-1] = 0;

		float r = 255;
		float g = 255;
		float b = 255;
		float a = defaultAlpha;

		sscanf(str, "%f %f %f %f", &r, &g, &b, &a);
		LogVerbose(LOG_TRT "class %02i  color %f %f %f %f\n", numLoaded, r, g, b, a);
		colors[numLoaded] = make_float4(r, g, b, a);
		numLoaded++; 
	}
	
	fclose(f);
	return validateClassColors(colors, numLoaded, expectedClasses, defaultAlpha);
}


// LoadClassColors
bool tensorNet::LoadClassColors( const char* filename, float4** colors, int expectedClasses, float defaultAlpha )
{
	// validate parameters
	if( !colors || expectedClasses <= 0 )
	{
		LogError(LOG_TRT "tensorNet::LoadClassColors() had invalid/NULL parameters\n");
		return false;
	}
	
	// allocate memory
	if( !cudaAllocMapped((void**)colors, expectedClasses * sizeof(float4)) )
		return false;
	
	// load colors
	return LoadClassColors(filename, colors[0], expectedClasses, defaultAlpha);
}


// GenerateColor
float4 tensorNet::GenerateColor( uint32_t classID, float alpha )
{
	// the first color is black, skip that one
	classID += 1;

	// https://github.com/dusty-nv/pytorch-segmentation/blob/16882772bc767511d892d134918722011d1ea771/datasets/sun_remap.py#L90
	#define bitget(byteval, idx)	((byteval & (1 << idx)) != 0)
	
	int r = 0;
	int g = 0;
	int b = 0;
	int c = classID;

	for( int j=0; j < 8; j++ )
	{
		r = r | (bitget(c, 0) << 7 - j);
		g = g | (bitget(c, 1) << 7 - j);
		b = b | (bitget(c, 2) << 7 - j);
		c = c >> 3;
	}

	return make_float4(r, g, b, alpha);
}
