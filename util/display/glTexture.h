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
 
#ifndef __GL_TEXTURE_H__
#define __GL_TEXTURE_H__


#include "cudaUtility.h"
#include "cuda_gl_interop.h"


/**
 * OpenGL texture
 */
class glTexture
{
public:
	static glTexture* Create( uint32_t width, uint32_t height, uint32_t format, void* data=NULL );
	~glTexture();
	
	void Render( float x, float y );
	void Render( float x, float y, float width, float height );
	void Render( const float4& rect );
	
	inline uint32_t GetID() const		{ return mID; }
	inline uint32_t GetWidth() const	{ return mWidth; }
	inline uint32_t GetHeight() const	{ return mHeight; }
	inline uint32_t GetFormat() const	{ return mFormat; }
	inline uint32_t GetSize() const	{ return mSize; }
	
	void* MapCUDA();
	void  Unmap();
	
	bool UploadCPU( void* data );
	
private:
	glTexture();
	bool init(uint32_t width, uint32_t height, uint32_t format, void* data);
	
	uint32_t mID;
	uint32_t mDMA;
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mFormat;
	uint32_t mSize;
	
	cudaGraphicsResource* mInteropCUDA;
	void* mInteropHost;
	void* mInteropDevice;
};


#endif
