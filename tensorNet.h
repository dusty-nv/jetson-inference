/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __TENSOR_NET_H__
#define __TENSOR_NET_H__


#include "NvInfer.h"
#include "NvCaffeParser.h"

#include <sstream>


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
					  uint32_t maxBatchSize=2 );

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
					  uint32_t maxBatchSize=2 );

	/**
	 * Manually enable layer profiling times.	
	 */
	void EnableProfiler();

	/**
	 * Manually enable debug messages and synchronization.
	 */
	void EnableDebug();

	/**
	 * Manually disable FP16 for debugging purposes.
	 */
	void DisableFP16();

	/**
 	 * Query for half-precision FP16 support.
	 */
	inline bool HasFP16() const		{ return mEnableFP16; }

	
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
				    const std::vector<std::string>& outputs,
				    uint32_t maxBatchSize, std::ostream& modelStream);
				
	/**
	 * Prefix used for tagging printed log output
	 */
	#define LOG_GIE "[GIE]  "
	
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

	nvinfer1::IRuntime* mInfer;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IExecutionContext* mContext;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mInputSize;
	float*   mInputCPU;
	float*   mInputCUDA;
	uint32_t mMaxBatchSize;
	bool	 mEnableProfiler;
	bool     mEnableDebug;
	bool	 mEnableFP16;
	bool     mOverride16;
	
	nvinfer1::Dims3 mInputDims;
	
	struct outputLayer
	{
		std::string name;
		nvinfer1::Dims3 dims;
		uint32_t size;
		float* CPU;
		float* CUDA;
	};
	
	std::vector<outputLayer> mOutputs;
};

#endif
