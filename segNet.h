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
 
#ifndef __SEGMENTATION_NET_H__
#define __SEGMENTATION_NET_H__


#include "tensorNet.h"


/**
 * Name of default input blob for segmentation model.
 * @ingroup deepVision
 */
#define SEGNET_DEFAULT_INPUT   "data"

/**
 * Name of default output blob for segmentation model.
 * @ingroup deepVision
 */
#define SEGNET_DEFAULT_OUTPUT  "score_fr_21classes"


/**
 * Image segmentation with FCN-Alexnet or custom models, using TensorRT.
 * @ingroup deepVision
 */
class segNet : public tensorNet
{
public:
	/**
	 * Enumeration of pretrained/built-in network models.
	 */
	enum NetworkType
	{
		FCN_ALEXNET_PASCAL_VOC,		    /**< FCN-Alexnet trained on Pascal VOC dataset. */
		FCN_ALEXNET_SYNTHIA_CVPR16,	    /**< FCN-Alexnet trained on SYNTHIA CVPR16 dataset. @note To save disk space, this model isn't downloaded by default. Enable it in CMakePreBuild.sh */
		FCN_ALEXNET_SYNTHIA_SUMMER_HD,    /**< FCN-Alexnet trained on SYNTHIA SEQS summer datasets. @note To save disk space, this model isn't downloaded by default. Enable it in CMakePreBuild.sh */
		FCN_ALEXNET_SYNTHIA_SUMMER_SD,    /**< FCN-Alexnet trained on SYNTHIA SEQS summer datasets. @note To save disk space, this model isn't downloaded by default. Enable it in CMakePreBuild.sh */
		FCN_ALEXNET_CITYSCAPES_HD,	    /**< FCN-Alexnet trained on Cityscapes dataset with 21 classes. */
		FCN_ALEXNET_CITYSCAPES_SD,	    /**< FCN-Alexnet trained on Cityscapes dataset with 21 classes. @note To save disk space, this model isn't downloaded by default. Enable it in CMakePreBuild.sh */
		FCN_ALEXNET_AERIAL_FPV_720p, 	    /**< FCN-Alexnet trained on aerial first-person view of the horizon line for drones, 1280x720 and 21 output classes */
		
		/* add new models here */
		SEGNET_CUSTOM
	};

	/**
 	 * Enumeration of mask/overlay filtering modes.
	 */
	enum FilterMode
	{
		FILTER_POINT,		/**< Single point sampling */
		FILTER_LINEAR		/**< Bilinear filtering */
	};

	/**
	 * Parse a string from one of the built-in pretrained models.
	 * Valid names are "cityscapes-hd", "cityscapes-sd", "pascal-voc", ect.
	 * @returns one of the segNet::NetworkType enums, or segNet::CUSTOM on invalid string.
	 */
	static NetworkType NetworkTypeFromStr( const char* model_name );

	/**
	 * Parse a string from one of the FilterMode values.
	 * Valid strings are "point", and "linear"
	 * @returns one of the segNet::FilterMode enums, or default segNet::FILTER_LINEAR on an error.
	 */
	static FilterMode FilterModeFromStr( const char* str, FilterMode default_value=FILTER_LINEAR );

	/**
	 * Load a new network instance
	 */
	static segNet* Create( NetworkType networkType=FCN_ALEXNET_CITYSCAPES_SD, uint32_t maxBatchSize=2,
					   precisionType precision=TYPE_FASTEST, deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param class_labels File path to list of class name labels
	 * @param class_colors File path to list of class colors
	 * @param input Name of the input layer blob. @see SEGNET_DEFAULT_INPUT
	 * @param output Name of the output layer blob. @see SEGNET_DEFAULT_OUTPUT
	 * @param maxBatchSize The maximum batch size that the network will support and be optimized for.
	 */
	static segNet* Create( const char* prototxt_path, const char* model_path, 
					   const char* class_labels, const char* class_colors=NULL,
					   const char* input = SEGNET_DEFAULT_INPUT, 
					   const char* output = SEGNET_DEFAULT_OUTPUT,
					   uint32_t maxBatchSize=2, precisionType precision=TYPE_FASTEST, 
					   deviceType device=DEVICE_GPU, bool allowGPUFallback=true );
	

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static segNet* Create( int argc, char** argv );
	
	/**
	 * Destroy
	 */
	virtual ~segNet();
	
	/**
 	 * Perform the initial inferencing processing portion of the segmentation.
	 * The results can then be visualized using the Overlay() and Mask() functions.
	 * @param input float4 input image in CUDA device memory, RGBA colorspace with values 0-255.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param ignore_class label name of class to ignore in the classification (or NULL to process all).
	 */
	bool Process( float* input, uint32_t width, uint32_t height, const char* ignore_class="void" );

	/**
	 * Produce a grayscale binary segmentation mask, where the pixel values
	 * correspond to the class ID of the corresponding class type.
	 */
	bool Mask( uint8_t* output, uint32_t width, uint32_t height );

	/**
	 * Produce a colorized RGBA segmentation mask.
	 */
	bool Mask( float* output, uint32_t width, uint32_t height, FilterMode filter=FILTER_LINEAR );

	/**
	 * Produce the segmentation overlay alpha blended on top of the original image.
	 * @param input float4 input image in CUDA device memory, RGBA colorspace with values 0-255.
	 * @param output float4 output image in CUDA device memory, RGBA colorspace with values 0-255.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param ignore_class label name of class to ignore in the classification (or NULL to process all).
	 * @param type overlay visualization options
	 * @returns true on success, false on error.
	 */
	bool Overlay( float* output, uint32_t width, uint32_t height, FilterMode filter=FILTER_LINEAR );

	/**
	 * Find the ID of a particular class (by label name).
	 */
	int FindClassID( const char* label_name );

	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const						{ return DIMS_C(mOutputs[0].dims); }

	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassLabel( uint32_t id )	const		{ return id < mClassLabels.size() ? mClassLabels[id].c_str() : NULL; }
	
	/**
	 * Retrieve the class synset category of a particular class.
	 */
	inline float* GetClassColor( uint32_t id ) const				{ return mClassColors[0] + (id*4); }

	/**
	 * Set the visualization color of a particular class of object.
	 */
	void SetClassColor( uint32_t classIndex, float r, float g, float b, float a=255.0f );
	
	/**
 	 * Set a global alpha value for all classes (between 0-255),
	 * (optionally except for those that have been explicitly set).
	 */
	void SetGlobalAlpha( float alpha, bool explicit_exempt=true );

	/**
 	 * Retrieve the path to the file containing the class label descriptions.
	 */
	inline const char* GetClassPath() const						{ return mClassPath.c_str(); }

	/**
	 * Retrieve the number of columns in the classification grid.
	 * This indicates the resolution of the raw segmentation output.
	 */
	inline uint32_t GetGridWidth() const						{ return DIMS_W(mOutputs[0].dims); }

	/**
	 * Retrieve the number of rows in the classification grid.
	 * This indicates the resolution of the raw segmentation output.
	 */
	inline uint32_t GetGridHeight() const						{ return DIMS_H(mOutputs[0].dims); }

	/**
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ return (mNetworkType != SEGNET_CUSTOM ? "FCN_Alexnet" : "segNet"); }

protected:
	segNet();
	
	bool classify( const char* ignore_class );

	bool overlayPoint( float* input, uint32_t in_width, uint32_t in_height, float* output, uint32_t out_width, uint32_t out_height, bool mask );
	bool overlayLinear( float* input, uint32_t in_width, uint32_t in_height, float* output, uint32_t out_width, uint32_t out_height, bool mask );
	
	bool loadClassColors( const char* filename );
	bool loadClassLabels( const char* filename );
	
	std::vector<std::string> mClassLabels;
	std::string mClassPath;

	float*   mClassColors[2];	/**< array of overlay colors in shared CPU/GPU memory */
	uint8_t* mClassMap[2];		/**< runtime buffer for the argmax-classified class index of each tile */
	
	float*   mLastInputImg;		/**< last input image to be processed, stored for overlay */
	uint32_t mLastInputWidth;	/**< width in pixels of last input image to be processed */
	uint32_t mLastInputHeight;	/**< height in pixels of last input image to be processed */

	NetworkType mNetworkType;	/**< Pretrained built-in model type enumeration */
};


#endif

