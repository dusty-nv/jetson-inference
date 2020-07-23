/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __HOMOGRAPHY_NET_H__
#define __HOMOGRAPHY_NET_H__

#include "tensorNet.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name homographyNet
/// Homography estimation DNN for registering and aligning images.
/// @ingroup deepVision
//////////////////////////////////////////////////////////////////////////////////

///@{

#ifdef HAS_OPENCV
/**
 * Defines if homographyNet DNN is available on the system or not.
 * @note homographyNet requires OpenCV 3.0.0 or newer on the system,
 * and is only supported with TensorRT version 5.0 and newer, as it 
 * uses ONNX models and requires ONNX importer support in TensorRT.
 * @ingroup homographyNet
 */
#define HAS_HOMOGRAPHY_NET
#endif

/**
 * Name of default input blob for homographyNet ONNX models.
 * @ingroup homographyNet
 */
#define HOMOGRAPHY_NET_DEFAULT_INPUT   "input_0"

/**
 * Name of default output blob for homographyNet ONNX models.
 * @ingroup homographyNet
 */
#define HOMOGRAPHY_NET_DEFAULT_OUTPUT  "output_0"


/**
 * Homography estimation networks with TensorRT support.
 * @ingroup homographyNet
 */
class homographyNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM = 0,	/**< Custom model from user */
		COCO_128,		/**< Synthetically-warped COCO (128x128 input) */
		WEBCAM_320	/**< Sequences collected from webcam (320x240 input) */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "coco", "coco_128", "coco-128", "webcam", "webcam_320", and "webcam-320".
	 * @returns one of the homographyNet::NetworkType enums, or homographyNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Load a new network instance
	 * @param networkType type of pre-supported network to load
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static homographyNet* Create( NetworkType networkType=WEBCAM_320, uint32_t maxBatchSize=1, 
						     precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, 
						     bool allowGPUFallback=true );
	
	/**
	 * Load a custom network instance
	 * @param model_path File path to the ONNX model.
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static homographyNet* Create( const char* model_path, 
						 	const char* input = HOMOGRAPHY_NET_DEFAULT_INPUT, 
						 	const char* output = HOMOGRAPHY_NET_DEFAULT_OUTPUT, 
							uint32_t maxBatchSize=1, precisionType precision=TYPE_FASTEST,
							deviceType device=DEVICE_GPU, bool allowGPUFallback=true );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static homographyNet* Create( int argc, char** argv );
	
	/**
	 * Destroy
	 */
	virtual ~homographyNet();
	
	/**
	 * Find the displacement from imageA to imageB.
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool FindDisplacement( float* imageA, float* imageB, uint32_t width, uint32_t height, float displacement[8] );
	
	/**
	 * Find the homography that warps imageA to imageB.
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool FindHomography( float* imageA, float* imageB, uint32_t width, uint32_t height, float H[3][3] );
	
	/**
	 * Find the homography (and it's inverse) that warps imageA to imageB.
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool FindHomography( float* imageA, float* imageB, uint32_t width, uint32_t height, float H[3][3], float H_inv[3][3] );

	/**
	 * Given the displacement from FindDisplacement(), compute the homography.
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool ComputeHomography( const float displacement[8], float H[3][3] );
	
	/**
	 * Given the displacement from FindDisplacement(), compute the homography and it's inverse.
	 * @returns True if the image was processed without error, false if an error was encountered.
	 */
	bool ComputeHomography( const float displacement[8], float H[3][3], float H_inv[3][3] );


protected:

	// constructor
	homographyNet();
};

///@}

#endif

