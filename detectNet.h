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
 
#ifndef __DETECT_NET_H__
#define __DETECT_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for detectNet model.
 * @ingroup deepVision
 */
#define DETECTNET_DEFAULT_INPUT   "data"

/**
 * Name of default output blob of the coverage map for detectNet model.
 * @ingroup deepVision
 */
#define DETECTNET_DEFAULT_COVERAGE  "coverage"

/**
 * Name of default output blob of the grid of bounding boxes for detectNet model.
 * @ingroup deepVision
 */
#define DETECTNET_DEFAULT_BBOX  "bboxes"


/**
 * Object recognition and localization networks with TensorRT support.
 * @ingroup deepVision
 */
class detectNet : public tensorNet
{
public:
	/**
	 * Object Detection result.
	 */
	struct Detection
	{
		// Object Info
		uint32_t ClassID;	/**< Class index of the detected object. */
		float Confidence;	/**< Confidence value of the detected object. */

		// Bounding Box Coordinates
		float Left;		/**< Left bounding box coordinate (in pixels) */
		float Right;		/**< Right bounding box coordinate (in pixels) */
		float Top;		/**< Top bounding box cooridnate (in pixels) */
		float Bottom;		/**< Bottom bounding box coordinate (in pixels) */

		/**< Calculate the width of the object */
		inline float Width() const						{ return Right - Left; }

		/**< Calculate the height of the object */
		inline float Height() const						{ return Bottom - Top; }

		/**< Calculate the area of the object */
		inline float Area() const						{ return Width() * Height(); }

		/**< Return the center of the object */
		inline void Center( float* x, float* y ) const		{ if(x) *x = Left + Width() * 0.5f; if(y) *y = Top + Height() * 0.5f; }

		/**< Return true if the coordinate is inside the bounding box */
		inline bool Contains( float x, float y ) const		{ return x >= Left && x <= Right && y >= Top && y <= Bottom; }

		/**< Return true if the bounding boxes overlap */
		inline bool Overlaps( const Detection& det ) const	{ return !(det.Left > Right || det.Right < Left || det.Top > Bottom || det.Bottom < Top); }
		
		/**< Return true if the bounding boxes overlap */
		inline bool Overlaps( float x1, float y1, float x2, float y2 ) const	{ return !(x1 > Right || x2 < Left || y1 > Bottom || y2 < Top); }
		
		/**< Expand the bounding box if they overlap (return true if so) */
		inline bool Expand( float x1, float y1, float x2, float y2 ) 	     { if(!Overlaps(x1, y1, x2, y2)) return false; Left = fminf(x1, Left); Top = fminf(y1, Top); Right = fmaxf(x2, Right); Bottom = fmaxf(y2, Bottom); return true; }
		
		/**< Expand the bounding box if they overlap (return true if so) */
		inline bool Expand( const Detection& det )      		{ if(!Overlaps(det)) return false; Left = fminf(det.Left, Left); Top = fminf(det.Top, Top); Right = fmaxf(det.Right, Right); Bottom = fmaxf(det.Bottom, Bottom); return true; }
		
		/**< Reset all member variables to zero */
		inline void Reset()								{ ClassID = 0; Confidence = 0; Left = 0; Right = 0; Top = 0; Bottom = 0; } 								
		
		/**< Default constructor */
		inline Detection()								{ Reset(); }
	};

		
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM = 0,		/**< Custom model from user */
		COCO_AIRPLANE,		/**< MS-COCO airplane class */
		COCO_BOTTLE,		/**< MS-COCO bottle class */
		COCO_CHAIR,		/**< MS-COCO chair class */
		COCO_DOG,			/**< MS-COCO dog class */
		FACENET,			/**< Human facial detector trained on FDDB */
		PEDNET,			/**< Pedestrian / person detector */
		PEDNET_MULTI		/**< Multi-class pedestrian + baggage detector */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "pednet", "multiped", "facenet", "face", "coco-airplane", "airplane",
	 * "coco-bottle", "bottle", "coco-chair", "chair", "coco-dog", or "dog".
	 * @returns one of the detectNet::NetworkType enums, or detectNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Load a new network instance
	 * @param networkType type of pre-supported network to load
	 * @param threshold default minimum threshold for detection
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static detectNet* Create( NetworkType networkType=PEDNET_MULTI, float threshold=0.5f, uint32_t maxBatchSize=2, 
						 precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a custom network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 * @param class_labels File path to list of class name labels
	 * @param threshold default minimum threshold for detection
	 * @param input Name of the input layer blob.
	 * @param coverage Name of the output coverage classifier layer blob, which contains the confidence values for each bbox.
	 * @param bboxes Name of the output bounding box layer blob, which contains a grid of rectangles in the image.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static detectNet* Create( const char* prototxt_path, const char* model_path, 
						 const char* mean_binary, const char* class_labels, float threshold=0.5f, 
						 const char* input = DETECTNET_DEFAULT_INPUT, 
						 const char* coverage = DETECTNET_DEFAULT_COVERAGE, 
						 const char* bboxes = DETECTNET_DEFAULT_BBOX,
						 uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST,
				   		 deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
							  
	/**
	 * Load a custom network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_pixel Input transform subtraction value (use 0.0 if the network already does this)
	 * @param class_labels File path to list of class name labels
	 * @param threshold default minimum threshold for detection
	 * @param input Name of the input layer blob.
	 * @param coverage Name of the output coverage classifier layer blob, which contains the confidence values for each bbox.
	 * @param bboxes Name of the output bounding box layer blob, which contains a grid of rectangles in the image.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static detectNet* Create( const char* prototxt_path, const char* model_path, float mean_pixel=0.0f, 
						 const char* class_labels=NULL, float threshold=0.5f, 
						 const char* input = DETECTNET_DEFAULT_INPUT, 
						 const char* coverage = DETECTNET_DEFAULT_COVERAGE, 
						 const char* bboxes = DETECTNET_DEFAULT_BBOX,
						 uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST,
				   		 deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static detectNet* Create( int argc, char** argv );
	
	/**
	 * Destory
	 */
	virtual ~detectNet();
	
	/**
	 * Detect object locations from an RGBA image, returning an array containing the detection results.
	 * @param[in]  input float4 RGBA input image in CUDA device memory.
	 * @param[in]  width width of the input image in pixels.
	 * @param[in]  height height of the input image in pixels.
	 * @param[out] detections pointer that will be set to array of detection results (residing in shared CPU/GPU memory)
	 * @param[in]  overlay if true, overlay the bounding boxes of the detected objects over the image (@see Overlay())
	 * @returns    The number of detected objects, 0 if there were no detected objects, and -1 if an error was encountered.
	 */
	int Detect( float* input, uint32_t width, uint32_t height, Detection** detections, bool overlay=true );

	/**
	 * Detect object locations in an RGBA image, into an array of the results allocated by the user.
	 * @param[in]  input float4 RGBA input image in CUDA device memory.
	 * @param[in]  width width of the input image in pixels.
	 * @param[in]  height height of the input image in pixels.
	 * @param[out] detections pointer to user-allocated array that will be filled with the detection results.
	 *                        @see GetMaxDetections() for the number of detection results that should be allocated in this buffer.
	 * @param[in]  overlay if true, overlay the bounding boxes of the detected objects over the image (@see Overlay())
	 * @returns    The number of detected objects, 0 if there were no detected objects, and -1 if an error was encountered.
	 */
	int Detect( float* input, uint32_t width, uint32_t height, Detection* detections, bool overlay=true );
	
	/**
	 * Draw the detected bounding boxes overlayed on an RGBA image.
	 * @note Overlay() will automatically be called by default by Detect(), if the overlay parameter is true 
	 * @param input float4 RGBA input image in CUDA device memory.
	 * @param output float4 RGBA output image in CUDA device memory.
	 * @param detections Array of detections allocated in CUDA device memory.
	 */
	bool Overlay( float* input, float* output, uint32_t width, uint32_t height, Detection* detections, uint32_t numDetections );
	
	/**
	 * Retrieve the minimum threshold for detection.
	 * TODO:  change this to per-class in the future
	 */
	inline float GetThreshold() const				{ return mCoverageThreshold; }

	/**
	 * Set the minimum threshold for detection.
	 */
	inline void SetThreshold( float threshold ) 		{ mCoverageThreshold = threshold; }

	/**
	 * Retrieve the maximum number of simultaneous detections the network supports.
	 * Knowing this is useful for allocating the buffers to store the output detection results.
	 */
	inline uint32_t GetMaxDetections() const		{ return DIMS_W(mOutputs[1].dims) * DIMS_H(mOutputs[1].dims) /** DIMS_C(mOutputs[1].dims)*/ * GetNumClasses(); }
		
	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const			{ return DIMS_C(mOutputs[0].dims); }

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
	 * Set the visualization color of a particular class of object.
	 */
	void SetClassColor( uint32_t classIndex, float r, float g, float b, float a=255.0f );
	
	
protected:

	// constructor
	detectNet();

	bool allocDetections();
	bool defaultColors();
	void defaultClassDesc();
	bool loadClassDesc( const char* filename );
	
	int clusterDetections( Detection* detections, uint32_t width, uint32_t height );

	float  mCoverageThreshold;
	float* mClassColors[2];
	float  mMeanPixel;

	std::vector<std::string> mClassDesc;
	std::vector<std::string> mClassSynset;

	std::string mClassPath;
	uint32_t    mCustomClasses;

	Detection* mDetectionSets[2];	// list of detections, NumDetectionSets * GetMaxDetections()
	uint32_t   mDetectionSet;		// index of next detection set to use
	static const uint32_t mNumDetectionSets = 16;	
};


#endif
