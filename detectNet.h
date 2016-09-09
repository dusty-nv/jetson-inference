/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __DETECT_NET_H__
#define __DETECT_NET_H__


#include "tensorNet.h"


/**
 * Object recognition and localization networks with GIE support.
 */
class detectNet : public tensorNet
{
public:
	/**
	 * Load a new network instance
	 * @param prototxt File path to the deployable network prototxt
	 * @param model File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 */
	static detectNet* Create( const char* prototxt, const char* model, const char* mean_binary,
							  const char* input_blob="data", const char* output_blob="bboxes" );
	
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
