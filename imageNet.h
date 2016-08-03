/*
 * inference-101
 */
 
#ifndef __IMAGE_NET_H__
#define __IMAGE_NET_H__


#include "caffeToGIE.h"


/**
 * Image recognition with GoogLeNet/Alexnet, using GIE.
 */
class imageNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		ALEXNET,
		GOOGLENET
	};

	/**
	 * Load a new network instance
	 */
	static imageNet* Create( NetworkType networkType=GOOGLENET );
	
	/**
	 * Destory
	 */
	~imageNet();
	
	/**
	 * Determine the maximum likelihood image class.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );

	/**
	 * Retrieve the number of image recognition classes (typically 1000)
	 */
	inline uint32_t GetNumClasses() const						{ return mOutputClasses; }
	
	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassDesc( uint32_t index )	const		{ return mClassDesc[index].c_str(); }
	
	/**
	 * Retrieve the class synset category of a particular class.
	 */
	inline const char* GetClassSynset( uint32_t index ) const		{ return mClassSynset[index].c_str(); }
	
	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ return (mNetworkType == GOOGLENET ? "googlenet" : "alexnet"); }

protected:
	imageNet();
	
	bool init( NetworkType networkType );
	bool loadClassInfo( const char* filename );
	
	nvinfer1::IRuntime* mInfer;
	nvinfer1::ICudaEngine* mEngine;
	nvinfer1::IExecutionContext* mContext;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mInputSize;
	float*   mInputCPU;
	float*   mInputCUDA;
	
	uint32_t mOutputSize;
	uint32_t mOutputClasses;
	float*   mOutputCPU;
	float*   mOutputCUDA;
	
	std::vector<std::string> mClassSynset;	// 1000 class ID's (ie n01580077, n04325704)
	std::vector<std::string> mClassDesc;

	NetworkType mNetworkType;
};


#endif
