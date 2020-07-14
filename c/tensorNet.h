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
 
#ifndef __TENSOR_NET_H__
#define __TENSOR_NET_H__

// forward declaration of IInt8Calibrator
namespace nvinfer1 { class IInt8Calibrator; }

// includes
#include <NvInfer.h>

#include <jetson-utils/cudaUtility.h>
#include <jetson-utils/commandLine.h>
#include <jetson-utils/imageFormat.h>
#include <jetson-utils/timespec.h>
#include <jetson-utils/logging.h>

#include <vector>
#include <sstream>
#include <math.h>


#if NV_TENSORRT_MAJOR > 5
typedef nvinfer1::Dims3 Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

#elif NV_TENSORRT_MAJOR > 1
typedef nvinfer1::DimsCHW Dims3;

#define DIMS_C(x) x.d[0]
#define DIMS_H(x) x.d[1]
#define DIMS_W(x) x.d[2]

#else
typedef nvinfer1::Dims3 Dims3; 

#define DIMS_C(x) x.c
#define DIMS_H(x) x.h
#define DIMS_W(x) x.w

#ifndef NV_TENSORRT_MAJOR
#define NV_TENSORRT_MAJOR 1
#define NV_TENSORRT_MINOR 0
#endif
#endif


/**
 * Default maximum batch size
 * @ingroup tensorNet
 */
#define DEFAULT_MAX_BATCH_SIZE  1

/**
 * Prefix used for tagging printed log output from TensorRT.
 * @ingroup tensorNet
 */
#define LOG_TRT "[TRT]    "


/**
 * Enumeration for indicating the desired precision that
 * the network should run in, if available in hardware.
 * @ingroup tensorNet
 */
enum precisionType
{
	TYPE_DISABLED = 0,	/**< Unknown, unspecified, or disabled type */
	TYPE_FASTEST,		/**< The fastest detected precision should be use (i.e. try INT8, then FP16, then FP32) */
	TYPE_FP32,		/**< 32-bit floating-point precision (FP32) */
	TYPE_FP16,		/**< 16-bit floating-point half precision (FP16) */
	TYPE_INT8,		/**< 8-bit integer precision (INT8) */
	NUM_PRECISIONS		/**< Number of precision types defined */
};

/**
 * Stringize function that returns precisionType in text.
 * @ingroup tensorNet
 */
const char* precisionTypeToStr( precisionType type );

/**
 * Parse the precision type from a string.
 * @ingroup tensorNet
 */
precisionType precisionTypeFromStr( const char* str );

/**
 * Enumeration for indicating the desired device that 
 * the network should run on, if available in hardware.
 * @ingroup tensorNet
 */
enum deviceType
{
	DEVICE_GPU = 0,			/**< GPU (if multiple GPUs are present, a specific GPU can be selected with cudaSetDevice() */
	DEVICE_DLA,				/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
	DEVICE_DLA_0 = DEVICE_DLA,	/**< Deep Learning Accelerator (DLA) Core 0 (only on Jetson Xavier) */
	DEVICE_DLA_1,				/**< Deep Learning Accelerator (DLA) Core 1 (only on Jetson Xavier) */
	NUM_DEVICES				/**< Number of device types defined */
};

/**
 * Stringize function that returns deviceType in text.
 * @ingroup tensorNet
 */
const char* deviceTypeToStr( deviceType type );

/**
 * Parse the device type from a string.
 * @ingroup tensorNet
 */
deviceType deviceTypeFromStr( const char* str );

/**
 * Enumeration indicating the format of the model that's
 * imported in TensorRT (either caffe, ONNX, or UFF).
 * @ingroup tensorNet
 */
enum modelType
{
	MODEL_CUSTOM = 0,	/**< Created directly with TensorRT API */
	MODEL_CAFFE,		/**< caffemodel */
	MODEL_ONNX,		/**< ONNX */
	MODEL_UFF,		/**< UFF */
	MODEL_ENGINE		/**< TensorRT engine/plan */
};

/**
 * Stringize function that returns modelType in text.
 * @ingroup tensorNet
 */
const char* modelTypeToStr( modelType type );

/**
 * Parse the model format from a string.
 * @ingroup tensorNet
 */
modelType modelTypeFromStr( const char* str );

/**
 * Parse the model format from a file path.
 * @ingroup tensorNet
 */
modelType modelTypeFromPath( const char* path );

/**
 * Profiling queries
 * @see tensorNet::GetProfilerTime()
 * @ingroup tensorNet
 */
enum profilerQuery
{
	PROFILER_PREPROCESS = 0,
	PROFILER_NETWORK,
	PROFILER_POSTPROCESS,
	PROFILER_VISUALIZE,
	PROFILER_TOTAL,
};

/**
 * Stringize function that returns profilerQuery in text.
 * @ingroup tensorNet
 */
const char* profilerQueryToStr( profilerQuery query );

/**
 * Profiler device
 * @ingroup tensorNet
 */
enum profilerDevice
{
	PROFILER_CPU = 0,	/**< CPU walltime */
	PROFILER_CUDA,		/**< CUDA kernel time */ 
};


/**
 * Abstract class for loading a tensor network with TensorRT.
 * For example implementations, @see imageNet and @see detectNet
 * @ingroup tensorNet
 */
class tensorNet
{
public:
	/**
	 * Destory
	 */
	virtual ~tensorNet();
	
	/**
	 * Load a new network instance
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blob The name of the input blob data to the network.
	 * @param output_blob The name of the output blob data from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean=NULL,
				   const char* input_blob="data", const char* output_blob="prob",
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Load a new network instance with multiple output layers
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blob The name of the input blob data to the network.
	 * @param output_blobs List of names of the output blobs from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean,
				   const char* input_blob, const std::vector<std::string>& output_blobs,
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Load a new network instance with multiple input layers.
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blobs List of names of the inputs blob data to the network.
	 * @param output_blobs List of names of the output blobs from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean,
				   const std::vector<std::string>& input_blobs, 
				   const std::vector<std::string>& output_blobs,
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
				   precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Load a new network instance (this variant is used for UFF models)
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blob The name of the input blob data to the network.
	 * @param input_dims The dimensions of the input blob (used for UFF).
	 * @param output_blobs List of names of the output blobs from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean,
				   const char* input_blob, const Dims3& input_dims, 
				   const std::vector<std::string>& output_blobs,
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
				   precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Load a new network instance with multiple input layers (used for UFF models)
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel 
	 * @param mean File path to the mean value binary proto (NULL if none)
	 * @param input_blobs List of names of the inputs blob data to the network.
	 * @param input_dims List of the dimensions of the input blobs (used for UFF).
	 * @param output_blobs List of names of the output blobs from the network.
	 * @param maxBatchSize The maximum batch size that the network will be optimized for.
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean,
				   const std::vector<std::string>& input_blobs, 
				   const std::vector<Dims3>& input_dims, 
				   const std::vector<std::string>& output_blobs,
				   uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
				   precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Load a network instance from a serialized engine plan file.
	 * @param engine_filename path to the serialized engine plan file.
	 * @param input_blobs List of names of the inputs blob data to the network.
	 * @param output_blobs List of names of the output blobs from the network.
	 */
	bool LoadEngine( const char* engine_filename,
				  const std::vector<std::string>& input_blobs, 
				  const std::vector<std::string>& output_blobs,
				  nvinfer1::IPluginFactory* pluginFactory=NULL,
				  deviceType device=DEVICE_GPU,
				  cudaStream_t stream=NULL );

	/**
	 * Load a network instance from a serialized engine plan file.
	 * @param engine_stream Memory containing the serialized engine plan file.
	 * @param engine_size Size of the serialized engine stream (in bytes).
	 * @param input_blobs List of names of the inputs blob data to the network.
	 * @param output_blobs List of names of the output blobs from the network.
	 */
	bool LoadEngine( char* engine_stream, size_t engine_size,
				  const std::vector<std::string>& input_blobs, 
				  const std::vector<std::string>& output_blobs,
				  nvinfer1::IPluginFactory* pluginFactory=NULL,
				  deviceType device=DEVICE_GPU,
				  cudaStream_t stream=NULL );

	/**
	 * Load network resources from an existing TensorRT engine instance.
	 * @param engine_stream Memory containing the serialized engine plan file.
	 * @param engine_size Size of the serialized engine stream (in bytes).
	 * @param input_blobs List of names of the inputs blob data to the network.
	 * @param output_blobs List of names of the output blobs from the network.
	 */
	bool LoadEngine( nvinfer1::ICudaEngine* engine,
				  const std::vector<std::string>& input_blobs, 
				  const std::vector<std::string>& output_blobs,
				  deviceType device=DEVICE_GPU,
				  cudaStream_t stream=NULL );

	/**
	 * Load a serialized engine plan file into memory.
	 */
	bool LoadEngine( const char* filename, char** stream, size_t* size );

	/**
	 * Manually enable layer profiling times.	
	 */
	void EnableLayerProfiler();

	/**
	 * Manually enable debug messages and synchronization.
	 */
	void EnableDebug();

	/**
 	 * Return true if GPU fallback is enabled.
	 */
	inline bool AllowGPUFallback() const					{ return mAllowGPUFallback; }

	/**
 	 * Retrieve the device being used for execution.
	 */
	inline deviceType GetDevice() const					{ return mDevice; }

	/**
	 * Retrieve the type of precision being used.
	 */
	inline precisionType GetPrecision() const				{ return mPrecision; }

	/**
	 * Check if a particular precision is being used.
	 */
	inline bool IsPrecision( precisionType type ) const		{ return (mPrecision == type); }

	/**
	 * Resolve a desired precision to a specific one that's available.
	 */
	static precisionType SelectPrecision( precisionType precision, deviceType device=DEVICE_GPU, bool allowInt8=true );

	/**
	 * Determine the fastest native precision on a device.
	 */
	static precisionType FindFastestPrecision( deviceType device=DEVICE_GPU, bool allowInt8=true );

	/**
	 * Detect the precisions supported natively on a device.
	 */
	static std::vector<precisionType> DetectNativePrecisions( deviceType device=DEVICE_GPU );
	
	/**
	 * Detect if a particular precision is supported natively.
	 */
	static bool DetectNativePrecision( const std::vector<precisionType>& nativeTypes, precisionType type );

	/**
	 * Detect if a particular precision is supported natively.
	 */
	static bool DetectNativePrecision( precisionType precision, deviceType device=DEVICE_GPU );

	/**
	 * Retrieve the stream that the device is operating on.
	 */
	inline cudaStream_t GetStream() const					{ return mStream; }

	/**
	 * Create and use a new stream for execution.
	 */
	cudaStream_t CreateStream( bool nonBlocking=true );

	/**
	 * Set the stream that the device is operating on.
	 */
	void SetStream( cudaStream_t stream );

	/**
	 * Retrieve the path to the network prototxt file.
	 */
	inline const char* GetPrototxtPath() const				{ return mPrototxtPath.c_str(); }

	/**
	 * Retrieve the path to the network model file.
	 */
	inline const char* GetModelPath() const					{ return mModelPath.c_str(); }

	/**
	 * Retrieve the format of the network model.
	 */
	inline modelType GetModelType() const					{ return mModelType; }

	/**
	 * Return true if the model is of the specified format.
	 */
	inline bool IsModelType( modelType type ) const			{ return (mModelType == type); }

	/**
	 * Retrieve the number of input layers to the network.
	 */
	inline uint32_t GetInputLayers() const					{ return mInputs.size(); }

	/**
	 * Retrieve the number of output layers to the network.
	 */
	inline uint32_t GetOutputLayers() const					{ return mOutputs.size(); }

	/**
	 * Retrieve the dimensions of network input layer.
	 */
	inline Dims3 GetInputDims( uint32_t layer=0 ) const		{ return mInputs[layer].dims; }

	/**
	 * Retrieve the width of network input layer.
	 */
	inline uint32_t GetInputWidth( uint32_t layer=0 ) const	{ return DIMS_W(mInputs[layer].dims); }

	/**
	 * Retrieve the height of network input layer.
	 */
	inline uint32_t GetInputHeight( uint32_t layer=0 ) const	{ return DIMS_H(mInputs[layer].dims); }

	/**
	 * Retrieve the size (in bytes) of network input layer.
	 */
	inline uint32_t GetInputSize( uint32_t layer=0 ) const		{ return mInputs[layer].size; }

	/**
	 * Retrieve the dimensions of network output layer.
	 */
	inline Dims3 GetOutputDims( uint32_t layer=0 ) const		{ return mOutputs[layer].dims; }

	/**
	 * Retrieve the width of network output layer.
	 */
	inline uint32_t GetOutputWidth( uint32_t layer=0 ) const	{ return DIMS_W(mOutputs[layer].dims); }

	/**
	 * Retrieve the height of network output layer.
	 */
	inline uint32_t GetOutputHeight( uint32_t layer=0 ) const	{ return DIMS_H(mOutputs[layer].dims); }

	/**
	 * Retrieve the size (in bytes) of network output layer.
	 */
	inline uint32_t GetOutputSize( uint32_t layer=0 ) const	{ return mOutputs[layer].size; }

	/**
	 * Retrieve the network frames per second (FPS).
	 */
	inline float GetNetworkFPS()							{ return 1000.0f / GetNetworkTime(); }

	/**
	 * Retrieve the network runtime (in milliseconds).
	 */
	inline float GetNetworkTime()							{ return GetProfilerTime(PROFILER_NETWORK, PROFILER_CUDA); }
	
	/**
	 * Retrieve the profiler runtime (in milliseconds).
	 */
	inline float2 GetProfilerTime( profilerQuery query )		{ PROFILER_QUERY(query); return mProfilerTimes[query]; }
	
	/**
	 * Retrieve the profiler runtime (in milliseconds).
	 */
	inline float GetProfilerTime( profilerQuery query, profilerDevice device ) { PROFILER_QUERY(query); return (device == PROFILER_CPU) ? mProfilerTimes[query].x : mProfilerTimes[query].y; }
	
	/**
	 * Print the profiler times (in millseconds).
	 */
	inline void PrintProfilerTimes()
	{
		LogInfo("\n");
		LogInfo(LOG_TRT "------------------------------------------------\n");
		LogInfo(LOG_TRT "Timing Report %s\n", GetModelPath());
		LogInfo(LOG_TRT "------------------------------------------------\n");

		for( uint32_t n=0; n <= PROFILER_TOTAL; n++ )
		{
			const profilerQuery query = (profilerQuery)n;

			if( PROFILER_QUERY(query) )
				LogInfo(LOG_TRT "%-12s  CPU %9.5fms  CUDA %9.5fms\n", profilerQueryToStr(query), mProfilerTimes[n].x, mProfilerTimes[n].y);
		}

		LogInfo(LOG_TRT "------------------------------------------------\n\n");

		static bool first_run=true;

		if( first_run )
		{
			LogWarning(LOG_TRT "note -- when processing a single image, run 'sudo jetson_clocks' before\n"
				      "                to disable DVFS for more accurate profiling/timing measurements\n\n");
			
			first_run = false;
		}
	}

protected:

	/**
	 * Constructor.
	 */
	tensorNet();
		
	/**
	 * Execute processing of the network.
	 * @param sync if true (default), the device will be synchronized after processing
	 *             and the thread/function will block until processing is complete. 
	 *             if false, the function will return immediately after the processing
	 *             has been enqueued to the CUDA stream indicated by GetStream().
	 */
	bool ProcessNetwork( bool sync=true );
	  
	/**
	 * Create and output an optimized network model
	 * @note this function is automatically used by LoadNetwork, but also can 
	 *       be used individually to perform the network operations offline.
	 * @param deployFile name for network prototxt
	 * @param modelFile name for model
	 * @param outputs network outputs
	 * @param maxBatchSize maximum batch size 
	 * @param modelStream output model stream
	 */
	bool ProfileModel( const std::string& deployFile, const std::string& modelFile,
				    const std::vector<std::string>& inputs, const std::vector<Dims3>& inputDims,
				    const std::vector<std::string>& outputs, uint32_t maxBatchSize, 
				    precisionType precision, deviceType device, bool allowGPUFallback,
				    nvinfer1::IInt8Calibrator* calibrator, char** engineStream, size_t* engineSize );

	/**
	 * Configure builder options
	 */
	bool ConfigureBuilder( nvinfer1::IBuilder* builder, uint32_t maxBatchSize, 
					   uint32_t workspaceSize, precisionType precision, 
					   deviceType device, bool allowGPUFallback, 
					   nvinfer1::IInt8Calibrator* calibrator );

	/**
	 * Logger class for GIE info/warning/errors
	 */
	class Logger : public nvinfer1::ILogger			
	{
	public:
		void log( Severity severity, const char* msg ) override
		{
			if( severity == Severity::kWARNING )
			{
				LogWarning(LOG_TRT "%s\n", msg);
			}
			else if( severity == Severity::kINFO )
			{
				LogInfo(LOG_TRT "%s\n", msg);
			}
		#if NV_TENSORRT_MAJOR > 5
			else if( severity == Severity::kVERBOSE )
			{
				LogVerbose(LOG_TRT "%s\n", msg);
			}
		#endif
			else
			{
				LogError(LOG_TRT "%s\n", msg);
			}
		}
	} static gLogger;

	/**
	 * Profiler interface for measuring layer timings
	 */
	class Profiler : public nvinfer1::IProfiler
	{
	public:
		Profiler() : timingAccumulator(0.0f)	{ }
		
		virtual void reportLayerTime(const char* layerName, float ms)
		{
			LogVerbose(LOG_TRT "layer %s - %f ms\n", layerName, ms);
			timingAccumulator += ms;
		}
		
		float timingAccumulator;
	} gProfiler;

	/**
	 * Begin a profiling query, before network is run
	 */
	inline void PROFILER_BEGIN( profilerQuery query )		
	{ 
		const uint32_t evt = query*2; 
		const uint32_t flag = (1 << query);

		CUDA(cudaEventRecord(mEventsGPU[evt], mStream)); 
		timestamp(&mEventsCPU[evt]); 

		mProfilerQueriesUsed |= flag;
		mProfilerQueriesDone &= ~flag;
	}

	/**
	 * End a profiling query, after the network is run
	 */
	inline void PROFILER_END( profilerQuery query )		
	{ 
		const uint32_t evt = query*2+1; 

		CUDA(cudaEventRecord(mEventsGPU[evt])); 
		timestamp(&mEventsCPU[evt]); 
		timespec cpuTime; 
		timeDiff(mEventsCPU[evt-1], mEventsCPU[evt], &cpuTime);
		mProfilerTimes[query].x = timeFloat(cpuTime);

		if( mEnableProfiler && query == PROFILER_NETWORK ) 
		{ 
			LogVerbose(LOG_TRT "layer network time - %f ms\n", gProfiler.timingAccumulator); 
			gProfiler.timingAccumulator = 0.0f; 
			LogWarning(LOG_TRT "note -- when processing a single image, run 'sudo jetson_clocks' before\n"
				      "                to disable DVFS for more accurate profiling/timing measurements\n"); 
		}
	}
	
	/**
	 * Query the CUDA part of a profiler query.
	 */
	inline bool PROFILER_QUERY( profilerQuery query )
	{
		const uint32_t flag = (1 << query);

		if( query == PROFILER_TOTAL )
		{
			mProfilerTimes[PROFILER_TOTAL].x = 0.0f;
			mProfilerTimes[PROFILER_TOTAL].y = 0.0f;

			for( uint32_t n=0; n < PROFILER_TOTAL; n++ )
			{
				if( PROFILER_QUERY((profilerQuery)n) )
				{
					mProfilerTimes[PROFILER_TOTAL].x += mProfilerTimes[n].x;
					mProfilerTimes[PROFILER_TOTAL].y += mProfilerTimes[n].y;
				}
			}

			return true;
		}
		else if( mProfilerQueriesUsed & flag )
		{
			if( !(mProfilerQueriesDone & flag) )
			{
				const uint32_t evt = query*2;
				float cuda_time = 0.0f;
				CUDA(cudaEventElapsedTime(&cuda_time, mEventsGPU[evt], mEventsGPU[evt+1]));
				mProfilerTimes[query].y = cuda_time;
				mProfilerQueriesDone |= flag;
				//mProfilerQueriesUsed &= ~flag;
			}

			return true;
		}

		return false;
	}

protected:

	/* Member Variables */
	std::string mPrototxtPath;
	std::string mModelPath;
	std::string mMeanPath;
	std::string mCacheEnginePath;
	std::string mCacheCalibrationPath;

	deviceType    mDevice;
	precisionType mPrecision;
	modelType     mModelType;
	cudaStream_t  mStream;
	cudaEvent_t   mEventsGPU[PROFILER_TOTAL * 2];
	timespec      mEventsCPU[PROFILER_TOTAL * 2];

	nvinfer1::IRuntime* mInfer;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IExecutionContext* mContext;
	
	float2   mProfilerTimes[PROFILER_TOTAL + 1];
	uint32_t mProfilerQueriesUsed;
	uint32_t mProfilerQueriesDone;
	uint32_t mWorkspaceSize;
	uint32_t mMaxBatchSize;
	bool	    mEnableProfiler;
	bool     mEnableDebug;
	bool	    mAllowGPUFallback;
	void**   mBindings;

	struct layerInfo
	{
		std::string name;
		Dims3 dims;
		uint32_t size;
		uint32_t binding;
		float* CPU;
		float* CUDA;
	};
	
	std::vector<layerInfo> mInputs;
	std::vector<layerInfo> mOutputs;
};

#endif
