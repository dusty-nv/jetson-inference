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

// include TensorRT
#include "NvInfer.h"

#include <vector>
#include <sstream>


#if NV_TENSORRT_MAJOR > 1
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
 */
#define MAX_BATCH_SIZE_DEFAULT  2


/**
 * Enumeration for indicating the desired precision that
 * the network should run in, if available in hardware.
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
 */
const char* precisionTypeToStr( precisionType type );

/**
 * Parse the precision type from a string.
 */
precisionType precisionTypeFromStr( const char* str );

/**
 * Enumeration for indicating the desired device that 
 * the network should run on, if available in hardware.
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
 */
const char* deviceTypeToStr( deviceType type );

/**
 * Parse the device type from a string.
 */
deviceType deviceTypeFromStr( const char* str );

/**
 * Enumeration indicating the format of the model that's
 * imported in TensorRT (either caffe, ONNX, or UFF).
 */
enum modelFormat
{
	MODEL_CUSTOM = 0,	/**< Created directly with TensorRT API */
	MODEL_CAFFE,		/**< caffemodel */
	MODEL_ONNX,		/**< ONNX */
	MODEL_UFF			/**< UFF */
};

/**
 * Stringize function that returns modelFormat in text.
 */
const char* modelFormatToStr( modelFormat format );

/**
 * Parse the model format from a string.
 */
modelFormat modelFormatFromStr( const char* str );


/**
 * Abstract class for loading a tensor network with TensorRT.
 * For example implementations, @see imageNet and @see detectNet
 * @ingroup deepVision
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
				   uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST,
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
				   uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST,
				   deviceType device=DEVICE_GPU, bool allowGPUFallback=true,
				   nvinfer1::IInt8Calibrator* calibrator=NULL, cudaStream_t stream=NULL );

	/**
	 * Manually enable layer profiling times.	
	 */
	void EnableProfiler();

	/**
	 * Manually enable debug messages and synchronization.
	 */
	void EnableDebug();

	/**
 	 * Return true if GPU fallback is enabled.
	 */
	inline bool AllowGPUFallback() const				{ return mAllowGPUFallback; }

	/**
 	 * Retrieve the device being used for execution.
	 */
	inline deviceType GetDevice() const				{ return mDevice; }

	/**
	 * Retrieve the type of precision being used.
	 */
	inline precisionType GetPrecision() const			{ return mPrecision; }

	/**
	 * Check if a particular precision is being used.
	 */
	inline bool IsPrecision( precisionType type ) const	{ return (mPrecision == type); }

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
	inline cudaStream_t GetStream() const				{ return mStream; }

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
	inline const char* GetPrototxtPath() const			{ return mPrototxtPath.c_str(); }

	/**
	 * Retrieve the path to the network model file.
	 */
	inline const char* GetModelPath() const				{ return mModelPath.c_str(); }

	/**
	 * Retrieve the format of the network model.
	 */
	inline modelFormat GetModelFormat() const			{ return mModelFormat; }

protected:

	/**
	 * Constructor.
	 */
	tensorNet();
			  
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
				    const std::vector<std::string>& outputs, uint32_t maxBatchSize, 
				    precisionType precision, deviceType device, bool allowGPUFallback,
				    nvinfer1::IInt8Calibrator* calibrator, std::ostream& modelStream);
				
	/**
	 * Prefix used for tagging printed log output
	 */
	#define LOG_GIE "[TRT]  "
	#define LOG_TRT LOG_GIE

	/**
	 * Logger class for GIE info/warning/errors
	 */
	class Logger : public nvinfer1::ILogger			
	{
		void log( Severity severity, const char* msg ) override
		{
			if( severity != Severity::kINFO /*|| mEnableDebug*/ )
				printf(LOG_GIE "%s\n", msg);
		}
	} gLogger;

	/**
	 * Profiler interface for measuring layer timings
	 */
	class Profiler : public nvinfer1::IProfiler
	{
	public:
		Profiler() : timingAccumulator(0.0f)	{ }
		
		virtual void reportLayerTime(const char* layerName, float ms)
		{
			printf(LOG_GIE "layer %s - %f ms\n", layerName, ms);
			timingAccumulator += ms;
		}
		
		float timingAccumulator;
		
	} gProfiler;

	/**
	 * When profiling is enabled, end a profiling section and report timing statistics.
	 */
	inline void PROFILER_REPORT()		{ if(mEnableProfiler) { printf(LOG_GIE "layer network time - %f ms\n", gProfiler.timingAccumulator); gProfiler.timingAccumulator = 0.0f; } }

protected:

	/* Member Variables */
	std::string mPrototxtPath;
	std::string mModelPath;
	std::string mMeanPath;
	std::string mInputBlobName;
	std::string mCacheEnginePath;
	std::string mCacheCalibrationPath;

	deviceType    mDevice;
	precisionType mPrecision;
	modelFormat   mModelFormat;
	cudaStream_t  mStream;
	cudaEvent_t   mEvents[2];

	nvinfer1::IRuntime* mInfer;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IExecutionContext* mContext;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mInputSize;
	float*   mInputCPU;
	float*   mInputCUDA;
	uint32_t mMaxBatchSize;
	bool	    mEnableProfiler;
	bool     mEnableDebug;
	bool	    mAllowGPUFallback;

	Dims3 mInputDims;
	
	struct outputLayer
	{
		std::string name;
		Dims3 dims;
		uint32_t size;
		float* CPU;
		float* CUDA;
	};
	
	std::vector<outputLayer> mOutputs;
};

#endif
