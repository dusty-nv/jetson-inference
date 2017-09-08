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
