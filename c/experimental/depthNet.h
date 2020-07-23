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
 
#ifndef __DEPTH_NET_H__
#define __DEPTH_NET_H__

#include "tensorNet.h"
#include "cudaColormap.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name depthNet
/// Depth estimation from monocular images.
/// @ingroup deepVision
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Name of default input blob for depthNet model.
 * @ingroup depthNet
 */
#define DEPTHNET_DEFAULT_INPUT   "input_0"

/**
 * Name of default output blob for depthNet model.
 * @ingroup depthNet
 */
#define DEPTHNET_DEFAULT_OUTPUT  "output_0"


/**
 * Command-line options able to be passed to depthNet::Create()
 * @ingroup depthNet
 */
#define DEPTHNET_USAGE_STRING  "depthNet arguments: \n" 							\
		  "  --network NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * TODO\n"                                         \
		  "  --model MODEL        path to custom model to load (onnx)\n" 			\
		  "  --input_blob INPUT   name of the input layer (default is '" DEPTHNET_DEFAULT_INPUT "')\n" 	\
		  "  --output_blob OUTPUT name of the output layer (default is '" DEPTHNET_DEFAULT_OUTPUT "')\n" 	\
		  "  --batch_size BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n"


/**
 * Depth estimation from monocular images, using TensorRT.
 * @ingroup depthNet
 */
class depthNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,        /**< Custom model provided by the user */
		MOBILENET,	/**< MobileNet backbone */
		RESNET_18,	/**< ResNet-18 backbone */
		RESNET_50,	/**< ResNet-50 backbone */
		FCN_RESNET_50,	/**< Fully-convolutional ResNet-50 */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "mobilenet", "resnet-18", or "resnet-50", ect.
	 * @returns one of the depthNet::NetworkType enums, or depthNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance
	 */
	static depthNet* Create( NetworkType networkType=MOBILENET, uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto (can be NULL)
	 * @param class_labels File path to list of class name labels
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static depthNet* Create( const char* model_path, 
						const char* input=DEPTHNET_DEFAULT_INPUT, 
						const char* output=DEPTHNET_DEFAULT_OUTPUT, 
						uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a custom network instance of a UFF model
	 * @param model_path File path to the UFF model
	 * @param input Name of the input layer blob.
	 * @param inputDims Dimensions of the input layer blob.
	 * @param output Name of the output layer blob containing the bounding boxes, ect.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static depthNet* Create( const char* model_path, const char* input,
						const Dims3& inputDims, const char* output,
						uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						precisionType precision=TYPE_FASTEST,
				   		deviceType device=DEVICE_GPU, bool allowGPUFallback=true );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static depthNet* Create( int argc, char** argv );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return DEPTHNET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~depthNet();
	
	/**
	 * Compute the depth field from a monocular RGBA image.
	 * @note the raw depth field can be retrieved with GetDepthField().
	 */
	bool Process( float* input, uint32_t width, uint32_t height );

	/**
	 * Process an RGBA image and map the depth image with the specified colormap.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* input, float* output, uint32_t width, uint32_t height, 
			    cudaColormapType colormap=COLORMAP_DEFAULT,
			    cudaFilterMode filter=FILTER_LINEAR );

	/**
	 * Process an RGBA image and map the depth image with the specified colormap.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* input, uint32_t input_width, uint32_t input_height,
			    float* output, uint32_t output_width, uint32_t output_height, 
			    cudaColormapType colormap=COLORMAP_DEFAULT,
			    cudaFilterMode filter=FILTER_LINEAR );

	/**
	 * Visualize the raw depth field into a colorized RGBA depth map.
	 * @note Visualize() should only be called after Process()
	 */
	bool Visualize( float* depth_map, uint32_t width, uint32_t height,
				 cudaColormapType colormap=COLORMAP_DEFAULT, 
				 cudaFilterMode filter=FILTER_LINEAR );

	/**
	 * Extract and save the point cloud to a PCD file (depth only).
	 * @note SavePointCloud() should only be called after Process()
	 */
	bool SavePointCloud( const char* filename );

	/**
	 * Extract and save the point cloud to a PCD file (depth + RGB).
	 * @note SavePointCloud() should only be called after Process()
	 */
	bool SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height );

	/**
	 * Extract and save the point cloud to a PCD file (depth + RGB).
	 * @note SavePointCloud() should only be called after Process()
	 */
	bool SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height,
					 const float2& focalLength, const float2& principalPoint );

	/**
	 * Extract and save the point cloud to a PCD file (depth + RGB).
	 * @note SavePointCloud() should only be called after Process()
	 */
	bool SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height,
					 const float intrinsicCalibration[3][3] );

	/**
	 * Extract and save the point cloud to a PCD file (depth + RGB).
	 * @note SavePointCloud() should only be called after Process()
	 */
	bool SavePointCloud( const char* filename, float* rgba, uint32_t width, uint32_t height,
					 const char* intrinsicCalibrationPath );

	/**
	 * Return the raw depth field.
	 */
	inline float* GetDepthField() const						{ return mOutputs[0].CUDA; }

	/**
	 * Return the width of the depth field.
	 */
	inline uint32_t GetDepthFieldWidth() const					{ return DIMS_W(mOutputs[0].dims); }

	/**
	 * Return the height of the depth field
	 */
	inline uint32_t GetDepthFieldHeight() const					{ return DIMS_H(mOutputs[0].dims); }

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ NetworkTypeToStr(mNetworkType); }

protected:
	depthNet();
	
	NetworkType mNetworkType;
	float2      mDepthRange;
};

///@}

#endif

