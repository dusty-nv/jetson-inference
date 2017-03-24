/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#ifndef __SEGMENTATION_NET_H__
#define __SEGMENTATION_NET_H__


#include "tensorNet.h"


/**
 * Image segmentation with FCN-Alexnet or custom models, using TensorRT.
 * @ingroup deepVision
 */
class segNet : public tensorNet
{
public:
	/**
	 * Network model enumeration.
	 */
	enum NetworkType
	{
		FCN_ALEXNET_PASCAL_VOC,		    /**< FCN-Alexnet trained on Pascal VOC dataset. */
		FCN_ALEXNET_SYNTHIA_CVPR16,	    /**< FCN-Alexnet trained on SYNTHIA CVPR16 dataset. */
		FCN_ALEXNET_SYNTHIA_SUMMER_HD,    /**< FCN-Alexnet trained on SYNTHIA SEQS summer datasets. */
		FCN_ALEXNET_SYNTHIA_SUMMER_SD,    /**< FCN-Alexnet trained on SYNTHIA SEQS summer datasets. */
		FCN_ALEXNET_CITYSCAPES_HD,	    /**< FCN-Alexnet trained on Cityscapes dataset with 21 classes. */
		FCN_ALEXNET_CITYSCAPES_SD,	    /**< FCN-Alexnet trained on Cityscapes dataset with 21 classes. */
		FCN_ALEXNET_AERIAL_FPV_720p_4ch,  /**< FCN-Alexnet trained on aerial first-person view of the horizon line for drones, 1280x720 and 4 output classes */ 
		FCN_ALEXNET_AERIAL_FPV_720p_21ch, /**< FCN-Alexnet trained on aerial first-person view of the horizon line for drones, 1280x720 and 21 output classes */
		FCN_ALEXNET_AERIAL_FPV_720p = FCN_ALEXNET_AERIAL_FPV_720p_21ch,
		
		/* add new models here */
		SEGNET_CUSTOM
	};

	/**
	 * Load a new network instance
	 */
	static segNet* Create( NetworkType networkType=FCN_ALEXNET_CITYSCAPES_SD );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param class_labels File path to list of class name labels
	 * @param class_colors File path to list of class colors
	 * @param input Name of the input layer blob.
	 * @param output Name of the output layer blob.
	 */
	static segNet* Create( const char* prototxt_path, const char* model_path, 
					   const char* class_labels, const char* class_colors=NULL,
					   const char* input="data", const char* output=/*"upscore_21classes"*/"score_fr_21classes" );
	

	/**
	 * Load a new network instance by parsing the command line.
	 */
	static segNet* Create( int argc, char** argv );
	
	/**
	 * Destroy
	 */
	virtual ~segNet();
	
	/**
	 * Produce the segmentation overlay alpha blended on top of the original image.
	 * @param input float4 input image in CUDA device memory, RGBA colorspace with values 0-255.
	 * @param output float4 output image in CUDA device memory, RGBA colorspace with values 0-255.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param alpha alpha blending value indicating transparency of the overlay.
	 * @param ignore_class label name of class to ignore in the classification (or NULL to process all).
	 * @returns true on success, false on error.
	 */
	bool Overlay( float* input, float* output, uint32_t width, uint32_t height, const char* ignore_class="void" );
	
	/**
	 * Find the ID of a particular class (by label name).
	 */
	int FindClassID( const char* label_name );

	/**
	 * Retrieve the number of object classes supported in the detector
	 */
	inline uint32_t GetNumClasses() const						{ return mOutputs[0].dims.c; }
	
	/**
	 * Retrieve the description of a particular class.
	 */
	inline const char* GetClassLabel( uint32_t id )	const		{ return mClassLabels[id].c_str(); }
	
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
	 * Retrieve the network type (alexnet or googlenet)
	 */
	inline NetworkType GetNetworkType() const					{ return mNetworkType; }

	/**
 	 * Retrieve a string describing the network name.
	 */
	inline const char* GetNetworkName() const					{ return (mNetworkType != SEGNET_CUSTOM ? "FCN_Alexnet" : "segNet"); }

protected:
	segNet();
	
	bool loadClassColors( const char* filename );
	bool loadClassLabels( const char* filename );
	
	std::vector<std::string> mClassLabels;
	float*   mClassColors[2];	/**< array of overlay colors in shared CPU/GPU memory */
	uint8_t* mClassMap[2];		/**< runtime buffer for the argmax-classified class index of each tile */

	NetworkType mNetworkType;
};


#endif

