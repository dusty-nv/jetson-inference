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
 * Default value of the scale used for drawing features
 */
#define FEATURENET_DEFAULT_DRAWING_SCALE 0.005f

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


// forward declarations
class cudaFont;


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
	 * Perform feature matching on a pair of images, returning a set of corresponding keypoints.
	 *
	 * This function takes as input two images, and outputs two arrays of matching feature coordinates,
	 * along with their confidence values.  Only matches that exceed the confidence threshold will be considered.
	 * If sorted is set to true, the corresponding keypoint lists will be sorted by confidence in descending order.
	 *
	 * @param image_A first input image in CUDA device memory.
	 * @param width_A width of the first input image (in pixels).
	 * @param height_A height of the first input image (in pixels).
	 * @param image_B second input image in CUDA device memory.
	 * @param width_B width of the second input image (in pixels).
	 * @param height_B height of the second input image (in pixels).
	 * @param features_A pointer to output array of matching keypoint coordinates from the first image.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long.
	 * @param features_B pointer to output array of matching keypoint coordinates from the second image.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long. 
	 * @param confidence pointer to output array of confidence values of the feature matches.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long. 
	 * @param threshold confidence threshold, below which matches are ignored (default is 0.01).
	 * @param sorted if true (default), the matches are sorted by confidence value in descending order.
	 *
	 * @returns The number of feature matches, or -1 if there was an error.
	 */
	template<typename T> int Match( T* image_A, uint32_t width_A, uint32_t height_A, 
							  T* image_B, uint32_t width_B, uint32_t height_B, 
							  float2* features_A, float2* features_B, float* confidence, 
							  float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true )	{ return Match(image_A, width_A, height_A, imageFormatFromType<T>(), image_B, width_B, height_B, imageFormatFromType<T>(), features_A, features_B, confidence, threshold, sorted); }
		
	/**
	 * Perform feature matching on a pair of images, returning a set of corresponding keypoints.
	 *
	 * This function takes as input two images, and outputs two arrays of matching feature coordinates,
	 * along with their confidence values.  Only matches that exceed the confidence threshold will be considered.
	 * If sorted is set to true, the corresponding keypoint lists will be sorted by confidence in descending order.
	 *
	 * This overload of Match() allocates the output memory arrays for the user.  This memory is owned by the 
	 * featureNet object and shouldn't be released by the user.  This memory can be re-used by featureNet on 
	 * future invocations of Match(), so if the user wishes to retain these features they should allocate their 
	 * own arrays and use the other version of Match() above.  It will be allocated as shared CPU/GPU memory.
	 *
	 * @param image_A first input image in CUDA device memory.
	 * @param width_A width of the first input image (in pixels).
	 * @param height_A height of the first input image (in pixels).
	 * @param image_B second input image in CUDA device memory.
	 * @param width_B width of the second input image (in pixels).
	 * @param height_B height of the second input image (in pixels).
	 * @param features_A output pointer that gets set to the array of matching keypoint coordinates from the first image.  
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param features_B pointer to output array of matching keypoint coordinates from the second image.
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param confidence pointer to output array of confidence values of the feature matches.
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param threshold confidence threshold, below which matches are ignored (default is 0.01).
	 * @param sorted if true (default), the matches are sorted by confidence value in descending order.
	 *
	 * @returns The number of feature matches, or -1 if there was an error.
	 */
	template<typename T> int Match( T* image_A, uint32_t width_A, uint32_t height_A, 
							  T* image_B, uint32_t width_B, uint32_t height_B, 
							  float2** features_A, float2** features_B, float** confidence, 
							  float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true )	{ return Match(image_A, width_A, height_A, imageFormatFromType<T>(), image_B, width_B, height_B, imageFormatFromType<T>(), features_A, features_B, confidence, threshold, sorted); }
			 			 
	/**
	 * Perform feature matching on a pair of images, returning a set of corresponding keypoints.
	 *
	 * This function takes as input two images, and outputs two arrays of matching feature coordinates,
	 * along with their confidence values.  Only matches that exceed the confidence threshold will be considered.
	 * If sorted is set to true, the corresponding keypoint lists will be sorted by confidence in descending order.
	 *
	 * @param image_A first input image in CUDA device memory.
	 * @param width_A width of the first input image (in pixels).
	 * @param height_A height of the first input image (in pixels).
	 * @param format_A format of the first input image.
	 * @param image_B second input image in CUDA device memory.
	 * @param width_B width of the second input image (in pixels).
	 * @param height_B height of the second input image (in pixels).
	 * @param format_B format of the second input image.
	 * @param features_A pointer to output array of matching keypoint coordinates from the first image.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long.
	 * @param features_B pointer to output array of matching keypoint coordinates from the second image.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long. 
	 * @param confidence pointer to output array of confidence values of the feature matches.
	 *                   this array should be allocated by the user to be GetMaxFeatures() long. 
	 * @param threshold confidence threshold, below which matches are ignored (default is 0.01).
	 * @param sorted if true (default), the matches are sorted by confidence value in descending order.
	 *
	 * @returns The number of feature matches, or -1 if there was an error.
	 */
	int Match( void* image_A, uint32_t width_A, uint32_t height_A, imageFormat format_A, 
			 void* image_B, uint32_t width_B, uint32_t height_B, imageFormat format_B, 
			 float2* features_A, float2* features_B, float* confidence, 
			 float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true );
		
	/**
	 * Perform feature matching on a pair of images, returning a set of corresponding keypoints.
	 *
	 * This function takes as input two images, and outputs two arrays of matching feature coordinates,
	 * along with their confidence values.  Only matches that exceed the confidence threshold will be considered.
	 * If sorted is set to true, the corresponding keypoint lists will be sorted by confidence in descending order.
	 *
	 * This overload of Match() allocates the output memory arrays for the user.  This memory is owned by the 
	 * featureNet object and shouldn't be released by the user.  This memory can be re-used by featureNet on 
	 * future invocations of Match(), so if the user wishes to retain these features they should allocate their 
	 * own arrays and use the other version of Match() above.  It will be allocated as shared CPU/GPU memory.
	 *
	 * @param image_A first input image in CUDA device memory.
	 * @param width_A width of the first input image (in pixels).
	 * @param height_A height of the first input image (in pixels).
	 * @param format_A format of the first input image.
	 * @param image_B second input image in CUDA device memory.
	 * @param width_B width of the second input image (in pixels).
	 * @param height_B height of the second input image (in pixels).
	 * @param format_B format of the second input image.
	 * @param features_A output pointer that gets set to the array of matching keypoint coordinates from the first image.  
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param features_B pointer to output array of matching keypoint coordinates from the second image.
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param confidence pointer to output array of confidence values of the feature matches.
	 *                   see notes above about conditions to using this memory, as it isn't persistent and gets re-used by Match().
	 * @param threshold confidence threshold, below which matches are ignored (default is 0.01).
	 * @param sorted if true (default), the matches are sorted by confidence value in descending order.
	 *
	 * @returns The number of feature matches, or -1 if there was an error.
	 */
	int Match( void* image_A, uint32_t width_A, uint32_t height_A, imageFormat format_A, 
			 void* image_B, uint32_t width_B, uint32_t height_B, imageFormat format_B, 
			 float2** features_A, float2** features_B, float** confidence, 
			 float threshold=FEATURENET_DEFAULT_THRESHOLD, bool sorted=true );	 
			 
	/**
	 * DrawFeatures (in-place overlay)
	 */
	template<typename T> bool DrawFeatures( T* image, uint32_t width, uint32_t height,
									float2* features, uint32_t numFeatures, bool drawText=true, 
									float scale=FEATURENET_DEFAULT_DRAWING_SCALE,
									const float4& color=make_float4(0,255,0,255))			{ return DrawFeatures(image, width, height, imageFormatFromType<T>(), features, numFeatures, drawText, scale, color); }
		
	/**
	 * DrawFeatures (on a different output image)
	 */
	template<typename T> bool DrawFeatures( T* input, T* output, uint32_t width, uint32_t height,
									float2* features, uint32_t numFeatures, bool drawText=true, 
									float scale=FEATURENET_DEFAULT_DRAWING_SCALE,
									const float4& color=make_float4(0,255,0,255))			{ return DrawFeatures(input, output, width, height, imageFormatFromType<T>(), features, numFeatures, drawText, scale, color); }
				    		    
	/**
	 * DrawFeatures (in-place overlay)
	 */
	bool DrawFeatures( void* image, uint32_t width, uint32_t height, imageFormat format,
				    float2* features, uint32_t numFeatures, bool drawText=true, 
				    float scale=FEATURENET_DEFAULT_DRAWING_SCALE,
				    const float4& color=make_float4(0,255,0,255));
					
	/**
	 * DrawFeatures (on a different output image)
	 */
	bool DrawFeatures( void* input, void* output, uint32_t width, uint32_t height, imageFormat format,
				    float2* features, uint32_t numFeatures, bool drawText=true, 
				    float scale=FEATURENET_DEFAULT_DRAWING_SCALE,
				    const float4& color=make_float4(0,255,0,255));
		
	/**
	 * FindHomography
	 */
	bool FindHomography( float2* features_A, float2* features_B, uint32_t numFeatures, float H[3][3], float H_inv[3][3] ) const;
	
	/**
	 * Retrieve the maximum number of features (default is 1200)
	 */
	inline uint32_t GetMaxFeatures() const						{ return mMaxFeatures; } 
	
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
	int  postProcess();
	
	void* mResizedImg;
	
	uint32_t mInputWidth;
	uint32_t mInputHeight;
	uint32_t mMaxFeatures;

	float2* mOutputFeatures[2];
	float*  mOutputConfidence;
	
	static const int mCellResolution = 16;  // for LoFTR
	
	cudaFont* mFont;
	NetworkType mNetworkType;
};


#endif
