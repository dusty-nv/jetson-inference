/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __DETECT_NET_H__
#define __DETECT_NET_H__


#include "tensorNet.h"


/**
 * Object recognition and localization networks with TensorRT support.
 */
class detectNet : public tensorNet
{
public:
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 * @param input Name of the input layer blob.
	 * @param coverage Name of the output coverage classifier layer blob, which contains the confidence values for each bbox.
	 * @param bboxes Name of the output bounding box layer blob, which contains a grid of rectangles in the image.
	 */
	static detectNet* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
							  const char* input="data", const char* coverage="coverage", const char* bboxes="bboxes" );
	
	/**
	 * Destory
	 */
	virtual ~detectNet();
	
	/**
	 * Detect object locations in the image.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Detect( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );

protected:

	// constructor
	detectNet();
	
	float mCoverageThreshold;
};


#endif
