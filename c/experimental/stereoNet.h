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
 
#ifndef __STEREO_NET_H__
#define __STEREO_NET_H__


#include "tensorNet.h"
#include "cudaColormap.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name stereoNet
/// Stereo disparity depth estimation.
/// @ingroup deepVision
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Name of default input layer for left image.
 * @ingroup stereoNet
 */
#define STEREONET_DEFAULT_INPUT_LEFT  "left"

/**
 * Name of default input layer for right image.
 * @ingroup stereoNet
 */
#define STEREONET_DEFAULT_INPUT_RIGHT "right"

/**
 * Name of default output layer for stereo disparity.
 * @ingroup stereoNet
 */
#define STEREONET_DEFAULT_OUTPUT "disp"

/**
 * Command-line options able to be passed to stereoNet::Create()
 * @ingroup stereoNet
 */
#define STEREONET_USAGE_STRING  "stereoNet arguments: \n" 							\
		  "  --network NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * TODO\n"                                         


/**
 * Stereo disparity depth estimation, using TensorRT.
 * @ingroup stereoNet
 */
class stereoNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		NV_SMALL = 0,
		NV_TINY,

		RESNET18,
		RESNET18_2D,

		DEFAULT_NETWORK = RESNET18_2D
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "nvsmall", "nvtiny", "resnet18", "resnet18-2D", or "default".
	 * @returns one of the stereoNet::NetworkType enums, or stereoNet::DEFAULT_NETWORK on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance
	 */
	static stereoNet* Create( NetworkType networkType=DEFAULT_NETWORK, 
						 uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
						 precisionType precision=TYPE_FASTEST,
				   		 deviceType device=DEVICE_GPU, bool allowGPUFallback=true );

	/**
	 * Load a new network instance by parsing the command line.
	 */
	//static stereoNet* Create( int argc, char** argv );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return STEREONET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~stereoNet();
	
	/**
	 * Compute the depth field from a pair of stereo RGBA images.
	 * @note the raw depth field can be retrieved with GetDepthField().
	 */
	bool Process( float* left, float* right, uint32_t width, uint32_t height );

	/**
	 * Process stereo RGBA images and map the depth image with the specified colormap.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* left, float* right, float* output, 
			    uint32_t width, uint32_t height, 
			    cudaColormapType colormap=COLORMAP_DEFAULT,
			    cudaFilterMode filter=FILTER_LINEAR );

	/**
	 * Process stereo RGBA images and map the depth image with the specified colormap.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* left, float* right, uint32_t input_width, uint32_t input_height,
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
	stereoNet();

	bool init( NetworkType networkType, uint32_t maxBatchSize, 
			 precisionType precision, deviceType device, bool allowGPUFallback );
	
	NetworkType mNetworkType;
	float2      mDepthRange;
};

///@}

#endif

