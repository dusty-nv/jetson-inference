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
 * @ingroup util
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
	
	/**
	 * Draw font overlay onto image
	 */
	bool RenderOverlay( float4* input, float4* output, uint32_t width, uint32_t height, 
						const char* str, int x, int y, const float4& color=make_float4(0, 0, 0, 255));
						
	/**
	 * Draw font overlay onto image
	 */
	bool RenderOverlay( float4* input, float4* output, uint32_t width, uint32_t height, 
						const std::vector< std::pair< std::string, int2 > >& text,
						const float4& color=make_float4(0.0f, 0.0f, 0.0f, 255.0f));
	
protected:
	cudaFont();
	bool init( const char* bitmap_path );

	float4* mFontMapCPU;
	float4* mFontMapGPU;
	
	int mFontMapWidth;
	int mFontMapHeight;
	int2 mFontCellSize;
	
	short4* mCommandCPU;
	short4* mCommandGPU;
	int     mCmdEntries;
	
	static const uint32_t MaxCommands = 1024;
};

#endif
