/*
 * Copyright (c) 2012, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __ACTION_NET_H__
#define __ACTION_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for actionNet model.
 * @ingroup actionNet
 */
#define ACTIONNET_DEFAULT_INPUT   "data"

/**
 * Name of default output confidence values for actionNet model.
 * @ingroup actionNet
 */
#define ACTIONNET_DEFAULT_OUTPUT  "prob"


/**
 * Standard command-line options able to be passed to actionNet::Create()
 * @ingroup actionNet
 */
#define ACTIONNET_USAGE_STRING  "actionNet arguments: \n" 							\
		  "  --network=NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * alexnet\n" 								\
		  "                           * googlenet (default)\n" 					\
		  "                           * googlenet-12\n" 							\
		  "                           * resnet-18\n" 							\
		  "                           * resnet-50\n" 							\
		  "                           * resnet-101\n" 							\
		  "                           * resnet-152\n" 							\
		  "                           * vgg-16\n" 								\
		  "                           * vgg-19\n" 								\
		  "                           * inception-v4\n" 							\
		  "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" 			\
		  "  --prototxt=PROTOTXT  path to custom prototxt to load (for .caffemodel only)\n" 				\
		  "  --labels=LABELS      path to text file containing the labels for each class\n" 				\
		  "  --input-blob=INPUT   name of the input layer (default is '" ACTIONNET_DEFAULT_INPUT "')\n" 	\
		  "  --output-blob=OUTPUT name of the output layer (default is '" ACTIONNET_DEFAULT_OUTPUT "')\n" 	\
		  "  --batch-size=BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * Image recognition with classification networks, using TensorRT.
 * @ingroup actionNet
 */
class actionNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,        /**< Custom model provided by the user */
		RESNET_18,	/**< ResNet-18 trained on 1000-class ILSVRC15 */
		RESNET_34,	/**< ResNet-50 trained on 1000-class ILSVRC15 */
		RESNET_50,	/**< ResNet-50 trained on 1000-class ILSVRC15 */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "alexnet", "googlenet", "googlenet-12", or "googlenet_12", ect.
	 * @returns one of the actionNet::NetworkType enums, or actionNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance
	 */
	static actionNet* Create( NetworkType networkType=RESNET_18, uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto (can be NULL)
	 * @param class_labels File path to list of class name labels
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static actionNet* Create( const char* model_path, const char* class_labels, 
						const char* input=ACTIONNET_DEFAULT_INPUT, 
						const char* output=ACTIONNET_DEFAULT_OUTPUT, 
						uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static actionNet* Create( int argc, char** argv );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static actionNet* Create( const commandLine& cmdLine );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return ACTIONNET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~actionNet();
	
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	template<typename T> int Classify( T* image, uint32_t width, uint32_t height, uint32_t frameSkip=2, float* confidence=NULL )		{ return Classify((void*)image, width, height, imageFormatFromType<T>(), frameSkip, confidence); }
	
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( void* image, uint32_t width, uint32_t height, imageFormat format, uint32_t frameSkip, float* confidence=NULL );

	/**
	 * Retrieve the number of image recognition classes (typically 1000)
	 */
	inline uint32_t GetNumClasses() const						{ return mNumClasses; }
	
	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassDesc( uint32_t index )	const		{ return mClassDesc[index].c_str(); }

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
	inline const char* GetNetworkName() const					{ return NetworkTypeToStr(mNetworkType); }

protected:
	actionNet();
	
	int  Classify( float* confidence=NULL );
	bool PreProcess( void* image, uint32_t width, uint32_t height, imageFormat format );
	bool Process();

	bool init( const char* model_path, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool loadClassInfo( const char* filename, int expectedClasses=-1 );
	
	uint32_t mNumClasses;
	uint32_t mNumFrames;
	uint32_t mNumFramesStored;
	
	std::vector<std::string> mClassDesc;

	std::string mClassPath;
	NetworkType mNetworkType;
};


#endif
