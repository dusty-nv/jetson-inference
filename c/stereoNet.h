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
	 * Valid names are "mobilenet", "resnet-18", or "resnet-50", ect.
	 * @returns one of the depthNet::NetworkType enums, or depthNet::CUSTOM on invalid string.
	 */
	//static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	//static const char* NetworkTypeToStr( NetworkType network );

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
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	//inline const char* GetNetworkName() const					{ NetworkTypeToStr(mNetworkType); }

protected:
	stereoNet();

	bool init( NetworkType networkType, uint32_t maxBatchSize, 
			 precisionType precision, deviceType device, bool allowGPUFallback );
	
	NetworkType mNetworkType;
};


#endif
