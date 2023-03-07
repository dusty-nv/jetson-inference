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
 
#ifndef __ACTION_NET_H__
#define __ACTION_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for actionNet model.
 * @ingroup actionNet
 */
#define ACTIONNET_DEFAULT_INPUT   "input"

/**
 * Name of default output confidence values for actionNet model.
 * @ingroup actionNet
 */
#define ACTIONNET_DEFAULT_OUTPUT  "output"

/**
 * The model type for actionNet in data/networks/models.json
 * @ingroup actionNet
 */
#define ACTIONNET_MODEL_TYPE "action"

/**
 * Standard command-line options able to be passed to actionNet::Create()
 * @ingroup actionNet
 */
#define ACTIONNET_USAGE_STRING  "actionNet arguments: \n" 							\
		  "  --network=NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * resnet-18 (default)\n"						\
		  "                           * resnet-34\n" 							\
		  "  --model=MODEL        path to custom model to load (.onnx)\n" 			\
		  "  --labels=LABELS      path to text file containing the labels for each class\n" 				\
		  "  --input-blob=INPUT   name of the input layer (default is '" ACTIONNET_DEFAULT_INPUT "')\n" 	\
		  "  --output-blob=OUTPUT name of the output layer (default is '" ACTIONNET_DEFAULT_OUTPUT "')\n" 	\
		  "  --threshold=CONF     minimum confidence threshold for classification (default is 0.01)\n" 	\
		  "  --skip-frames=SKIP   how many frames to skip between classifications (default is 1)\n"         \
		  "  --profile            enable layer profiling in TensorRT\n\n"


/**
 * Action/activity classification on a sequence of images or video, using TensorRT.
 * @ingroup actionNet
 */
class actionNet : public tensorNet
{
public:
	/**
	 * Load a pre-trained model, either "resnet-18" or "resnet-34".
	 */
	static actionNet* Create( const char* network="resnet-18", uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
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
	 * Append an image to the sequence and classify the action, returning the index of the top class.
	 * Either the class with the maximum confidence will be returned, or -1 if no class meets 
	 * the threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * If this frame was skipped due to SetSkipFrames() being used, then the last frame's results will
	 * be returned.  By default, every other frame is skipped in order to lengthen the action's window.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum likelihood class, or -1 on error.
	 */
	template<typename T> int Classify( T* image, uint32_t width, uint32_t height, float* confidence=NULL )		{ return Classify((void*)image, width, height, imageFormatFromType<T>(), confidence); }
	
	/**
	 * Append an image to the sequence and classify the action, returning the index of the top class.
	 * Either the class with the maximum confidence will be returned, or -1 if no class meets 
	 * the threshold set by SetThreshold() or the `--threshold` command-line argument.
	 *
	 * If this frame was skipped due to SetSkipFrames() being used, then the last frame's results will
	 * be returned.  By default, every other frame is skipped in order to lengthen the action's window.
	 *
	 * @param image input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum likelihood class, or -1 on error.
	 */
	int Classify( void* image, uint32_t width, uint32_t height, imageFormat format, float* confidence=NULL );

	/**
	 * Retrieve the number of image recognition classes
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
	 * Return the number of frames that are skipped in between classifications.
	 * @see SetFrameSkip for more info.
	 */
	inline uint32_t GetSkipFrames() const						{ return mSkipFrames; }
	
	/**
	 * Set the number of frames that are skipped in between classifications.
	 * Since actionNet operates on video sequences, it's often helpful to skip frames 
	 * to lengthen the window of time the model gets to 'see' an action being performed.
	 *
	 * The default setting is 1, where every other frame is skipped.
	 * Setting this to 0 will disable it, and every frame will be processed.
	 * When a frame is skipped, the classification results from the last frame are returned.
	 */
	inline void SetSkipFrames( uint32_t frames )					{ mSkipFrames = frames; }
	 
protected:
	actionNet();
	
	bool init( const char* model_path, const char* class_path, const char* input, const char* output, uint32_t maxBatchSize, precisionType precision, deviceType device, bool allowGPUFallback );
	bool preProcess( void* image, uint32_t width, uint32_t height, imageFormat format );

	float* mInputBuffers[2];
	
	uint32_t mNumClasses;
	uint32_t mNumFrames;	// number of frames fed into the model
	uint32_t mSkipFrames;	// number of frames to skip when processing
	uint32_t mFramesSkipped;	// frame skip counter
	
	uint32_t mCurrentInputBuffer;
	uint32_t mCurrentFrameIndex;

	float mThreshold;
	float mLastConfidence;
	int   mLastClassification;
	
	std::vector<std::string> mClassDesc;

	std::string mClassPath;
};


#endif
