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
 
#ifndef __IMAGE_NET_H__
#define __IMAGE_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for imageNet model.
 * @ingroup deepVision
 */
#define IMAGENET_DEFAULT_INPUT   "data"

/**
 * Name of default output confidence values for imageNet model.
 * @ingroup deepVision
 */
#define IMAGENET_DEFAULT_OUTPUT  "prob"


/**
 * Image recognition with GoogleNet/Alexnet or custom models, using TensorRT.
 * @ingroup deepVision
 */
class imageNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,
		ALEXNET,		/**< 1000-class ILSVR12 */
		GOOGLENET,	/**< 1000-class ILSVR12 */
		GOOGLENET_12	/**< 12-class subset of ImageNet ILSVR12 from the tutorial */
	};


	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "alexnet", "googlenet", "googlenet-12", or "googlenet_12".
	 * @returns one of the imageNet::NetworkType enums, or imageNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );


	/**
	 * Load a new network instance
	 */
	static imageNet* Create( NetworkType networkType=GOOGLENET, uint32_t maxBatchSize=2, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto (can be NULL)
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static imageNet* Create( const char* prototxt_path, const char* model_path, 
						const char* mean_binary, const char* class_labels, 
						const char* input=IMAGENET_DEFAULT_INPUT, 
						const char* output=IMAGENET_DEFAULT_OUTPUT, 
						uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static imageNet* Create( int argc, char** argv );

	/**
	 * Destroy
	 */
	virtual ~imageNet();
	
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );

	/**
	 * Determine the maximum likelihood image class.
	 * @note before calling this function, you must call PreProcess() with the image. 
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* confidence=NULL );

	/**
	 * Perform pre-processing on the image to apply mean-value subtraction and
	 * to organize the data into NCHW format and BGR colorspace that the networks expect.
 	 * After calling PreProcess(), you can call Classify() without supplying all the parameters.
	 */
	bool PreProcess( float* rgba, uint32_t width, uint32_t height );

	/**
	 * Process the network, without determining the classification argmax.
	 * To perform the actual classification via post-processing, Classify() should be used instead.
	 */
	bool Process();

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
 	 * Retrieve the path to the file containing the class descriptions.
	 */
	inline const char* GetClassPath() const						{ return mClassPath.c_str(); }

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ if(mNetworkType == GOOGLENET) return "googlenet"; else if(mNetworkType == GOOGLENET_12) return "googlenet_12"; else if(mNetworkType == ALEXNET) return "alexnet"; else return "custom"; }

protected:
	imageNet();
	
	bool init( NetworkType networkType, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool loadClassInfo( const char* filename );
	
	uint32_t mCustomClasses;
	uint32_t mOutputClasses;
	
	std::vector<std::string> mClassSynset;	// 1000 class ID's (ie n01580077, n04325704)
	std::vector<std::string> mClassDesc;

	std::string mClassPath;
	NetworkType mNetworkType;
};


#endif
