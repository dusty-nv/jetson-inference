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
 * @ingroup imageNet
 */
#define IMAGENET_DEFAULT_INPUT   "data"

/**
 * Name of default output confidence values for imageNet model.
 * @ingroup imageNet
 */
#define IMAGENET_DEFAULT_OUTPUT  "prob"

/**
 * Default value of the minimum confidence threshold for classification.
 * @ingroup imageNet
 */
#define IMAGENET_DEFAULT_THRESHOLD 0.01f

/**
 * The model type for imageNet in data/networks/models.json
 * @ingroup imageNet
 */
#define IMAGENET_MODEL_TYPE "classification"

/**
 * Standard command-line options able to be passed to imageNet::Create()
 * @ingroup imageNet
 */
#define IMAGENET_USAGE_STRING  "imageNet arguments: \n" 							\
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
		  "  --input-blob=INPUT   name of the input layer (default is '" IMAGENET_DEFAULT_INPUT "')\n" 	\
		  "  --output-blob=OUTPUT name of the output layer (default is '" IMAGENET_DEFAULT_OUTPUT "')\n" 	\
		  "  --threshold=CONF     minimum confidence threshold for classification (default is 0.01)\n" 	\
		  "  --smoothing=WEIGHT   weight between [0,1] or number of frames (disabled by default)\n"		\
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * Image recognition with classification networks, using TensorRT.
 * @ingroup imageNet
 */
class imageNet : public tensorNet
{
public:
	/**
	 * List of classification results where each entry represents a (classID, confidence) pair.
	 */
	typedef std::vector<std::pair<uint32_t, float>> Classifications;
	
	/**
	 * Load one of the following pre-trained models:
	 *
	 *    - alexnet, googlenet, googlenet-12, 
	 *    - resnet-18, resnet-50, resnet-101, resnet-152, 
	 *    - vgg-16, vgg-19, inception-v4
	 *
	 * These are all 1000-class models trained on ImageNet ILSVRC,
	 * except for googlenet-12 which is a 12-class subset of ILSVRC.
	 */
	static imageNet* Create( const char* network="googlenet", 
						uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
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
	static imageNet* Create( const char* prototxt_path, const char* model_path, 
						const char* mean_binary, const char* class_labels, 
						const char* input=IMAGENET_DEFAULT_INPUT, 
						const char* output=IMAGENET_DEFAULT_OUTPUT, 
						uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static imageNet* Create( int argc, char** argv );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static imageNet* Create( const commandLine& cmdLine );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return IMAGENET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~imageNet();
			
	/**
	 * Predict the maximum-likelihood image class whose confidence meets the minimum threshold.
	 * Either the class with the maximum probability will be returned, or -1 if no class meets 
	 * the threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 *
	 * @returns ID of the class with the highest confidence, or -1 if no classes met the threshold.
      *          If a runtime error occurred during processing, then a value of -2 will be returned. 
	 */
	template<typename T> int Classify( T* image, uint32_t width, uint32_t height, float* confidence=NULL )		{ return Classify((void*)image, width, height, imageFormatFromType<T>(), confidence); }
	
	/**
	 * Predict the maximum-likelihood image class whose confidence meets the minimum threshold.
	 * Either the class with the maximum probability will be returned, or -1 if no class meets 
	 * the threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param format format of the image (rgb8, rgba8, rgb32f, rgba32f are supported)
	 * @param confidence optional pointer to float filled with confidence value.
	 *
	 * @returns ID of the class with the highest confidence, or -1 if no classes met the threshold.
      *          If a runtime error occurred during processing, then a value of -2 will be returned. 
	 */
	int Classify( void* image, uint32_t width, uint32_t height, imageFormat format, float* confidence=NULL );

	/**
	 * Predict the maximum-likelihood image class whose confidence meets the minimum threshold.
	 * Either the class with the maximum probability will be returned, or -1 if no class meets 
	 * the threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
      * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @param format format of the image (rgb8, rgba8, rgb32f, rgba32f are supported)
	 *
	 * @returns ID of the class with the highest confidence, or -1 if no classes met the threshold.
      *          If a runtime error occurred during processing, then a value of -2 will be returned. 
	 * @deprecated this overload of Classify() provides legacy compatibility with `float*` type (RGBA32F).
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL, imageFormat format=IMAGE_RGBA32F );

	/**
	 * Classify the image and return the topK image classification results that meet the minimum
	 * confidence threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param classifications returns a list of the topK (classID, confidence) classification resuts, sorted from highest to lowest confidence.
	 * @param topK the number of predictions to return (it can be less than this number if there weren't that many valid predictions)
	 *             The default value of topK is 1, in which case only the highest-confidence result wil be returned.
	 *             If the value of topK is <= 0, then all the valid predictions with confidence >= threshold will be returned.
	 *
	 * @returns ID of the class with the highest confidence, or -1 if no classes met the threshold.
      *          If a runtime error occurred during processing, then a value of -2 will be returned. 
	 */
	template<typename T> int Classify( T* image, uint32_t width, uint32_t height, Classifications& classifications, int topK=1 )		{ return Classify((void*)image, width, height, imageFormatFromType<T>(), classifications, topK); }

	/**
	 * Classify the image and return the topK image classification results that meet the minimum
	 * confidence threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param format format of the image (rgb8, rgba8, rgb32f, rgba32f are supported)
	 * @param classifications returns a list of the topK (classID, confidence) classification resuts, sorted from highest to lowest confidence.
	 * @param topK the number of predictions to return (it can be less than this number if there weren't that many valid predictions)
	 *             The default value of topK is 1, in which case only the highest-confidence result wil be returned.
	 *             If the value of topK is <= 0, then all the valid predictions with confidence >= threshold will be returned.
	 *
	 * @returns ID of the class with the highest confidence, or -1 if no classes met the threshold.
      *          If a runtime error occurred during processing, then a value of -2 will be returned. 
	 */
	int Classify( void* image, uint32_t width, uint32_t height, imageFormat format, Classifications& classifications, int topK=1 );

	/**
	 * Retrieve the number of image recognition classes (typically 1000)
	 */
	inline uint32_t GetNumClasses() const						{ return mNumClasses; }
	
	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassLabel( int index ) const			{ return GetClassDesc(index); }
	
	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassDesc( int index )	const			{ return index >= 0 ? mClassDesc[index].c_str() : "none"; }
	
	/**
	 * Retrieve the class synset category of a particular class.
	 */
	inline const char* GetClassSynset( int index ) const			{ return index >= 0 ? mClassSynset[index].c_str() : "none"; }
	
	/**
 	 * Retrieve the path to the file containing the class descriptions.
	 */
	inline const char* GetClassPath() const						{ return mClassPath.c_str(); }
	
	/**
	 * Return the confidence threshold used for classification.
	 */
	inline float GetThreshold() const							{ return mThreshold; }
	
	/**
	 * Set the confidence threshold used for classification.
	 * Classes with a confidence below this threshold will be ignored.
	 * @note this can also be set using the `--threshold=N` command-line argument.
	 */
	inline void SetThreshold( float threshold ) 					{ mThreshold = threshold; }
	
	/**
	 * Return the temporal smoothing weight or number of frames in the smoothing window.
	 * @see SetSmoothing
	 */
	inline float GetSmoothing() const							{ return mSmoothingFactor; }
	
	/**
	 * Enable temporal smoothing of the results using EWMA (exponentially-weighted moving average).
	 * This filters the confidence values of each class over ~N frames to reduce noise and jitter.
	 * In lieu of storing a history of past data, this uses an accumulated approximation of EMA:
	 *
	 *    EMA(x,t) = EMA(x, t-1) + w * (x - EMA(x, t-1))
	 *
	 * where x is a class softmax output logit, t is the timestep, and w is the smoothing weight.
	 *
	 * @param factor either a weight between [0,1] that's placed on the latest confidence values,
	 *               or the smoothing window as a number of frames (where the weight will be 1/N).  
	 *               For example, a factor of N=5 would average over approximately the last 5 frames,
	 *               and would be equivalent to specifying a weight of 0.2 (either can be used).
      *               A weight closer to 1 will be more responsive to changes, but also more noisy.	 
	 *               Setting this to 0 or 1 will disable smoothing and use the unfiltered outputs.
	 *
	 * @note this can also be set using the `--smoothing=N` command-line argument.
	 */
	inline void SetSmoothing( float factor ) 					{ mSmoothingFactor = factor; }

protected:
	imageNet();
	
	//bool init( NetworkType networkType, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool init(const char* prototxt_path, const char* model_path, const char* mean_binary, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool loadClassInfo( const char* filename, int expectedClasses=-1 );
	
	bool preProcess( void* image, uint32_t width, uint32_t height, imageFormat format );
	
	float* applySmoothing();
	
	uint32_t mNumClasses;
	
	std::vector<std::string> mClassSynset;	// 1000 class ID's (ie n01580077, n04325704)
	std::vector<std::string> mClassDesc;

	std::string mClassPath;
	//NetworkType mNetworkType;
	
	float* mSmoothingBuffer;
	float  mSmoothingFactor;
	
	float mThreshold;
};


#endif
