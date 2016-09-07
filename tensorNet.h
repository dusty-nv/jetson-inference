/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __TENSOR_NET_H__
#define __TENSOR_NET_H__


#include "NvInfer.h"
#include "NvCaffeParser.h"

#include <sstream>


/**
 * Abstract class for loading a tensor network with GIE.
 * For example implementations, @see imageNet and @see detectNet
 */
class tensorNet
{
public:
	/**
	 * Destory
	 */
	virtual ~tensorNet();
	
protected:

	/**
	 * Constructor.
	 */
	tensorNet();
	
	/**
	 * Load a new network instance
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel (NULL if none)
	 * @param mean File path to the mean value binary proto
	 */
	bool LoadNetwork( const char* prototxt, const char* model, const char* mean=NULL,
					  const char* input_blob="data", const char* output_blob="prob");

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
			if( severity != Severity::kINFO )
				printf(LOG_GIE "%s\n", msg);
		}
	} gLogger;

	
	/* Member Variables */
	std::string mPrototxtPath;
	std::string mModelPath;
	std::string mMeanPath;
	std::string mInputBlobName;
	std::string mOutputBlobName;
	
	nvinfer1::IRuntime* mInfer;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IExecutionContext* mContext;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mInputSize;
	float*   mInputCPU;
	float*   mInputCUDA;
	
	nvinfer1::Dims3 mInputDims;
	nvinfer1::Dims3 mOutputDims;
	
	uint32_t mOutputSize;
	uint32_t mOutputClasses;
	float*   mOutputCPU;
	float*   mOutputCUDA;
};

#endif
