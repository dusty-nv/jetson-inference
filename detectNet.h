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
	 * Detect object locations in the RGBA image.
	 * @param rgba float4 RGBA input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param numBoxes pointer to a single integer containing the maximum number of boxes available in boundingBoxes.
	 *                 upon successful return, will be set to the number of bounding boxes detected in the image.
	 * @param boundingBoxes pointer to array of bounding boxes.
	 * @param confidence optional pointer to float2 array filled with a (confidence, class) pair for each bounding box (numBoxes) 
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool Detect( float* rgba, uint32_t width, uint32_t height, float* boundingBoxes, int* numBoxes, float* confidence=NULL );
	
	/**
	 * Draw bounding boxes in the RGBA image.
	 * @param input float4 RGBA input image in CUDA device memory.
	 * @param output float4 RGBA output image in CUDA device memory.
	 */
	bool DrawBoxes( float* input, float* output, uint32_t width, uint32_t height, const float* boundingBoxes, int numBoxes, int classIndex=0 );
	
	/**
	 * Retrieve the maximum number of bounding boxes the network supports.
	 * Knowing this is useful for allocating the buffers to store the output bounding boxes.
	 */
	inline uint32_t GetMaxBoundingBoxes() const		{ return mOutputs[1].dims.w * mOutputs[1].dims.h * mOutputs[1].dims.c; }
		
	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const			{ return mOutputs[0].dims.c; }

	/**
	 * Set the visualization color of a particular class of object.
	 */
	void SetClassColor( uint32_t classIndex, float r, float g, float b, float a=1.0f );
	
	
protected:

	// constructor
	detectNet();
	
	float  mCoverageThreshold;
	float* mClassColors[2];
};


#endif
