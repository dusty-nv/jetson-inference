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
 
#ifndef __FLOW_NET_H__
#define __FLOW_NET_H__

#include "tensorNet.h"
#include "cudaColormap.h"


//////////////////////////////////////////////////////////////////////////////////
/// @name flowNet
/// Dense optical flow estimation DNN.
/// @ingroup deepVision
//////////////////////////////////////////////////////////////////////////////////

///@{

/**
 * Name of default input blob for flowNet model.
 * @ingroup flowNet
 */
#define FLOWNET_DEFAULT_INPUT   "input_0"

/**
 * Name of default output blob for flowNet model.
 * @ingroup flowNet
 */
#define FLOWNET_DEFAULT_OUTPUT  "output_0"


/**
 * Command-line options able to be passed to flowNet::Create()
 * @ingroup flowNet
 */
#define FLOWNET_USAGE_STRING  "flowNet arguments: \n" 							\
		  "  --network NETWORK    pre-trained model to load, one of the following:\n" 	\
		  "                           * TODO\n"                                         \
		  "  --model MODEL        path to custom model to load (onnx)\n" 			\
		  "  --input_blob INPUT   name of the input layer (default is '" FLOWNET_DEFAULT_INPUT "')\n" 	\
		  "  --output_blob OUTPUT name of the output layer (default is '" FLOWNET_DEFAULT_OUTPUT "')\n" 	\
		  "  --batch_size BATCH   maximum batch size (default is 1)\n"								\
		  "  --profile            enable layer profiling in TensorRT\n"


/**
 * Dense optical flow estimation, using TensorRT.
 * @ingroup flowNet
 */
class flowNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		CUSTOM,         /**< Custom model provided by the user */
		FLOW_320x240,	 /**< Dense optical flow model (320x240) */
		FLOW_512x384,	 /**< Dense optical flow model (512x384) */
		FLOW_640x480	 /**< Dense optical flow model (640x480) */
	};

	/**
	 * Parse a string to one of the built-in pretrained models.
	 * Valid names are "optical-flow-320x240", "flow-320x240", "flow-640x480", ect.
	 * @returns one of the flowNet::NetworkType enums, or flowNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Convert a NetworkType enum to a string.
	 */
	static const char* NetworkTypeToStr( NetworkType network );

	/**
	 * Load a new network instance
	 */
	static flowNet* Create( NetworkType networkType=FLOW_320x240, 
					    uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
					    precisionType precision=TYPE_FASTEST,
				   	    deviceType device=DEVICE_GPU, 
                             bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance
	 * @param model_path File path to the model (ONNX)
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static flowNet* Create( const char* model_path, 
					    const char* input=FLOWNET_DEFAULT_INPUT, 
					    const char* output=FLOWNET_DEFAULT_OUTPUT, 
					    uint32_t maxBatchSize=DEFAULT_MAX_BATCH_SIZE, 
					    precisionType precision=TYPE_FASTEST,
				   	    deviceType device=DEVICE_GPU, 
                             bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static flowNet* Create( int argc, char** argv );

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return FLOWNET_USAGE_STRING; }

	/**
	 * Destroy
	 */
	virtual ~flowNet();

	/**
	 * Compute the flow field from two sequential RGBA images.
	 * @note the flow field can be returned from GetFlowField()
	 */
	bool Process( float* prev_image, float* next_image, uint32_t width, uint32_t height );

	/**
	 * Process two sequential RGBA images and output the colorized flow map.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* prev_image, float* next_image, float* flow_map, 
			    uint32_t width, uint32_t height, cudaFilterMode filter=FILTER_LINEAR ); 

	/**
	 * Process two sequential RGBA images and output the colorized flow map.
	 * @note this function calls Process() followed by Visualize().
	 */
	bool Process( float* prev_image, float* next_image, uint32_t width, uint32_t height,
			    float* flow_map, uint32_t flow_map_width, uint32_t flow_map_height,
			    cudaFilterMode flow_map_filter=FILTER_LINEAR ); 

	/**
	 * Visualize the raw flow field into a colorized RGBA flow map.
	 */
	bool Visualize( float* flow_map, uint32_t width, uint32_t height, cudaFilterMode filter=FILTER_LINEAR );

	/**
	 * Return the x/y flow field in CHW format, where `C=2`.
      * The flow in the x-coordinate is contained in `C=0`, and
	 * the flow in the y-coordinate is contained in `C=1`.
	 * `H=GetFlowFieldHeight()` and `W=GetFlowFieldWidth()`.
	 */
	inline float* GetFlowField() const							{ return mOutputs[0].CUDA; }

	/**
	 * Return the width of the flow field.
	 */
	inline uint32_t GetFlowFieldWidth() const					{ return DIMS_W(mOutputs[0].dims); }

	/**
	 * Return the height of the flow field.
	 */
	inline uint32_t GetFlowFieldHeight() const					{ return DIMS_H(mOutputs[0].dims); }

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ NetworkTypeToStr(mNetworkType); }

protected:
	flowNet();
	
	NetworkType mNetworkType;
};

///@}

#endif

