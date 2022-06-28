/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __FEATURE_NET_H__
#define __FEATURE_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for featureNet model.
 * @ingroup featureNet
 */
#define FEATURENET_DEFAULT_INPUT_0   "img0"

/**
 * Name of default input blob for featureNet model.
 * @ingroup featureNet
 */
#define FEATURENET_DEFAULT_INPUT_1   "img1"

/**
 * Name of default output confidence values for featureNet model.
 * @ingroup featureNet
 */
#define FEATURENET_DEFAULT_OUTPUT  "output"

/**
 * Default value of the minimum detection threshold
 * @ingroup featureNet
 */
#define FEATURENET_DEFAULT_THRESHOLD 0.01f

/**
 * Standard command-line options able to be passed to featureNet::Create()
 * @ingroup featureNet
 */
#define FEATURENET_USAGE_STRING  "featureNet arguments: \n" 							\
		  "  --network=NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * loftr-coarse (default)\n"					\
		  "  --model=MODEL        path to custom model to load (caffemodel, uff, or onnx)\n" 			\
		  "  --input-blob=INPUT   name of the input layer (default is '" FEATURENET_DEFAULT_INPUT_0 "')\n" 	\
		  "  --output-blob=OUTPUT name of the output layer (default is '" FEATURENET_DEFAULT_OUTPUT "')\n" 	\
		  "  --batch-size=BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * Image recognition with classification networks, using TensorRT.
 * @ingroup featureNet
 */
class featureNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,        /**< Custom model provided by the user */
		LOFTR_COARSE,	/**< LoFTR coarse with distillation, trained on BlendedMVS */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "alexnet", "googlenet", "googlenet-12", or "googlenet_12", ect.
	 * @returns one of the featureNet::NetworkType enums, or featureNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance
	 */
	static featureNet* Create( NetworkType networkType=LOFTR_COARSE, uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						  precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, 
						  bool allowGPUFallback=true );
	
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
	static featureNet* Create( const char* model_path, 
						  const char* input_0=FEATURENET_DEFAULT_INPUT_0,
						  const char* input_1=FEATURENET_DEFAULT_INPUT_1, 						  
						  const char* output=FEATURENET_DEFAULT_OUTPUT, 
						  uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						  precisionType precision=TYPE_FASTEST,
				   		  deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static featureNet* Create( int argc, char** argv );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static featureNet* Create( const commandLine& cmdLine );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return FEATURENET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~featureNet();
	
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	//template<typename T> int Classify( T* imageA, T* imageB, uint32_t width, uint32_t height, float* confidence=NULL )		{ return Classify((void*)image, width, height, imageFormatFromType<T>(), confidence); }
	
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	//inline int Match( void* image_a, void* image_b, uint32_t width, uint32_t height, imageFormat format, 
	//		        float2* points_a, float2* points_b, float* confidence=NULL, float threshold=0.01 );

	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	//int Match( void* image_a, uint32_t width_a, uint32_t height_a, imageFormat format_a, 
	//		 void* image_b, uint32_t width_b, uint32_t height_b, imageFormat format_b, 
	//		 float2** points_a, float2** points_b, float** confidence=NULL, float threshold=0.01 );
			 
	/**
	 * Determine the maximum likelihood image class.
	 * This function performs pre-processing to the image (apply mean-value subtraction and NCHW format), @see PreProcess() 
	 * @param rgba input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Match( void* image_A, uint32_t width_A, uint32_t height_A, imageFormat format_A, 
			 void* image_B, uint32_t width_B, uint32_t height_B, imageFormat format_B, 
			 float2* keypoints_A, float2* keypoints_B, float* confidence=NULL, 
			 float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true );
			 
	/*int Match( void* images[2], uint32_t width[2], uint32_t height[2], imageFormat format[2], 
			 float2* features[2], float* confidence=NULL, 
			 float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true );*/
		
	/**
	 * Retrieve the maximum number of features (default is 1200)
	 */
	inline uint32_t GetMaxFeatures() const						{ return mOutputs[0].dims.d[1]; } 
	
	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ return NetworkTypeToStr(mNetworkType); }

protected:
	featureNet();
	
	bool init( const char* model_path, const char* input_0, const char* input_1, const char* output, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool preProcess( void* image, uint32_t width, uint32_t height, imageFormat format, uint32_t binding );

	void* mResizedImg;
	
	uint32_t mInputWidth;
	uint32_t mInputHeight;
	
	NetworkType mNetworkType;
};


#endif
