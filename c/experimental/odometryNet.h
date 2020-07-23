/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __ODOMETRY_NET_H__
#define __ODOMETRY_NET_H__


#include "tensorNet.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name odometryNet
/// Visual odometry estimation from image sequences.
/// @ingroup deepVision
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Name of default input blob for odometryNet ONNX models.
 * @ingroup odometryNet
 */
#define ODOMETRY_NET_DEFAULT_INPUT   "input_0"

/**
 * Name of default output blob for odometryNet ONNX models.
 * @ingroup odometryNet
 */
#define ODOMETRY_NET_DEFAULT_OUTPUT  "output_0"

/**
 * Command-line options able to be passed to odometryNet::Create()
 * @ingroup segNet
 */
#define ODOMETRY_NET_USAGE_STRING  "odometryNet arguments: \n" 						\
		  "  --network NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * resnet18-tum\n"							\
		  "                           * resnet18-cooridor\n"						\
		  "  --model MODEL        path to custom model to load (onnx)\n" 			\
		  "  --input_blob INPUT   name of the input layer (default: '" ODOMETRY_NET_DEFAULT_INPUT "')\n" 	\
		  "  --output_blob OUTPUT name of the output layer (default: '" ODOMETRY_NET_DEFAULT_OUTPUT "')\n" 	\
		  "  --batch_size BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n"								\
		  "  --verbose            enable verbose output from TensorRT\n"


/**
 * Visual odometry estimation networks with TensorRT support.
 * @ingroup odometryNet
 */
class odometryNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM = 0,		/**< Custom model from user */
		RESNET18_TUM,		/**< TUM RGB-D SLAM model using ResNet-18 (224x224 input) */
		RESNET18_COORIDOR	/**< Indoor cooridor model using ResNet-18 (224x224 input) */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "resnet18-tum", "tum", "resnet18-coordidor", and "cooridor".
	 * @returns one of the odometryNet::NetworkType enums, or odometryNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a human-readable string.
	 * @returns stringized version of the provided NetworkType enum.
	 */
	static const char* NetworkTypeToStr( NetworkType networkType );

	/**
	 * Load a new network instance
	 * @param networkType type of pre-supported network to load
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static odometryNet* Create( NetworkType networkType=RESNET18_TUM, uint32_t maxBatchSize=1, 
						   precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, 
						   bool allowGPUFallback=true );
	
	/**
	 * Load a custom network instance
	 * @param model_path File path to the ONNX model.
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static odometryNet* Create( const char* model_path, 
						   const char* input = ODOMETRY_NET_DEFAULT_INPUT, 
						   const char* output = ODOMETRY_NET_DEFAULT_OUTPUT, 
						   uint32_t maxBatchSize=1, precisionType precision=TYPE_FASTEST,
						   deviceType device=DEVICE_GPU, bool allowGPUFallback=true );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static odometryNet* Create( int argc, char** argv );
	
	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage()	{ return ODOMETRY_NET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~odometryNet();
	
	/**
	 * Process the odometry estimation from a pair of sequential images.
	 *
	 * @param imageA CUDA device pointer to the first RGBA input image in the sequence
	 * @param imageB CUDA device pointer to the second RGBA input image in the sequence
	 * @param width the width of the input images, in pixels
	 * @param height the height of the input images, in pixels
	 * @param output optional array that the output odometry data will get copied to.
	 *               GetNumOutputs() indicates the number of floats in the 1D array.
	 *               GetOutput() can retrieve a pointer to the data without copying.
	 *
	 * @returns true on success, or false if an error occurred
	 */
	bool Process( float4* imageA, float4* imageB, uint32_t width, uint32_t height, float* output=NULL );

	inline float* GetInput() const						{ return mInputs[0].CUDA; }

	/**
	 * Retrieve the pointer to the output odometry data.
	 */
	inline float* GetOutput() const						{ return mOutputs[0].CUDA; }

	/**
	 * Retrieve one of the odometry output data values.
	 */
	inline float GetOutput( uint32_t index ) const			{ return mOutputs[0].CPU[index]; }

	/**
	 * Retrieve the number of floats in the output odometry array.
	 */
	inline uint32_t GetNumOutputs() const					{ return DIMS_C(mOutputs[0].dims); }

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const				{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const				{ return NetworkTypeToStr(mNetworkType); }

protected:

	// constructor
	odometryNet();

	NetworkType mNetworkType;	/**< Pretrained built-in model type enumeration */
};

///@}

#endif

