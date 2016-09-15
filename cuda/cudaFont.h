/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __CUDA_FONT_H__
#define __CUDA_FONT_H__

#include "cudaUtility.h"

#include <string>
#include <vector>


/**
 * Font overlay rendering using CUDA
 */
class cudaFont
{
public:
	/**
	 * Create new CUDA font overlay object using textured fonts
	 */
	static cudaFont* Create( const char* font_bitmap="fontmapA.png" );
	
	/**
	 * Destructor
	 */
	~cudaFont();
	
protected:
	cudaFont();
	bool init( const char* bitmap_path );
	
	struct Element
	{
		std::string format;
		int2		position;
		int         argSlot;
	};
	
	float4* mFontMapCPU;
	float4* mFontMapGPU;
	
	int mFontMapWidth;
	int mFontMapHeight;
} ;


#endif