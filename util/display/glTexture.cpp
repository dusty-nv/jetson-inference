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

#include "glUtility.h"
#include "glTexture.h"

#include "cudaMappedMemory.h"


//-----------------------------------------------------------------------------------
inline uint32_t glTextureLayout( uint32_t format )
{
	switch(format)
	{
		case GL_LUMINANCE8:
		case GL_LUMINANCE16:			
		case GL_LUMINANCE32UI_EXT:
		case GL_LUMINANCE8I_EXT:
		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE16F_ARB:
		case GL_LUMINANCE32F_ARB:		return GL_LUMINANCE;

		case GL_LUMINANCE8_ALPHA8:		
		case GL_LUMINANCE16_ALPHA16:
		case GL_LUMINANCE_ALPHA32UI_EXT:
		case GL_LUMINANCE_ALPHA8I_EXT:
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB: return GL_LUMINANCE_ALPHA;
		
		case GL_RGB8:					
		case GL_RGB16:
		case GL_RGB32UI:
		case GL_RGB8I:
		case GL_RGB16I:
		case GL_RGB32I:
		case GL_RGB16F_ARB:
		case GL_RGB32F_ARB:				return GL_RGB;

		case GL_RGBA8:
		case GL_RGBA16:
		case GL_RGBA32UI:
		case GL_RGBA8I:
		case GL_RGBA16I:
		case GL_RGBA32I:
		//case GL_RGBA_FLOAT32:
		case GL_RGBA16F_ARB:
		case GL_RGBA32F_ARB:			return GL_RGBA;
	}

	return 0;
}


inline uint32_t glTextureLayoutChannels( uint32_t format )
{
	const uint layout = glTextureLayout(format);

	switch(layout)
	{
		case GL_LUMINANCE:			return 1;
		case GL_LUMINANCE_ALPHA:	return 2;
		case GL_RGB:				return 3;
		case GL_RGBA:				return 4;
	}

	return 0;
}


inline uint32_t glTextureType( uint32_t format )
{
	switch(format)
	{
		case GL_LUMINANCE8:
		case GL_LUMINANCE8_ALPHA8:
		case GL_RGB8:
		case GL_RGBA8:					return GL_UNSIGNED_BYTE;

		case GL_LUMINANCE16:
		case GL_LUMINANCE16_ALPHA16:
		case GL_RGB16:
		case GL_RGBA16:					return GL_UNSIGNED_SHORT;

		case GL_LUMINANCE32UI_EXT:
		case GL_LUMINANCE_ALPHA32UI_EXT:
		case GL_RGB32UI:
		case GL_RGBA32UI:				return GL_UNSIGNED_INT;

		case GL_LUMINANCE8I_EXT:
		case GL_LUMINANCE_ALPHA8I_EXT:
		case GL_RGB8I:
		case GL_RGBA8I:					return GL_BYTE;

		case GL_LUMINANCE16I_EXT:
		case GL_LUMINANCE_ALPHA16I_EXT:
		case GL_RGB16I:
		case GL_RGBA16I:				return GL_SHORT;

		case GL_LUMINANCE32I_EXT:
		case GL_LUMINANCE_ALPHA32I_EXT:
		case GL_RGB32I:
		case GL_RGBA32I:				return GL_INT;


		case GL_LUMINANCE16F_ARB:
		case GL_LUMINANCE_ALPHA16F_ARB:
		case GL_RGB16F_ARB:
		case GL_RGBA16F_ARB:			return GL_FLOAT;

		case GL_LUMINANCE32F_ARB:
		case GL_LUMINANCE_ALPHA32F_ARB:
		//case GL_RGBA_FLOAT32:
		case GL_RGB32F_ARB:
		case GL_RGBA32F_ARB:			return GL_FLOAT;
	}

	return 0;
}


inline uint glTextureTypeSize( uint32_t format )
{
	const uint type = glTextureType(format);

	switch(type)
	{
		case GL_UNSIGNED_BYTE:
		case GL_BYTE:					return 1;

		case GL_UNSIGNED_SHORT:
		case GL_SHORT:					return 2;

		case GL_UNSIGNED_INT:
		case GL_INT:
		case GL_FLOAT:					return 4;
	}

	return 0;
}
//-----------------------------------------------------------------------------------

// constructor
glTexture::glTexture()
{
	mID     = 0;
	mDMA    = 0;
	mWidth  = 0;
	mHeight = 0;
	mFormat = 0;
	mSize   = 0;
	
	mInteropCUDA   = NULL;
	mInteropHost   = NULL;
	mInteropDevice = NULL;
}


// destructor
glTexture::~glTexture()
{
	GL(glDeleteTextures(1, &mID));
}
	

// Create
glTexture* glTexture::Create( uint32_t width, uint32_t height, uint32_t format, void* data )
{
	glTexture* tex = new glTexture();
	
	if( !tex->init(width, height, format, data) )
	{
		printf("[OpenGL]  failed to create %ux%u texture\n", width, height);
		return NULL;
	}
	
	return tex;
}
		
		
// Alloc
bool glTexture::init( uint32_t width, uint32_t height, uint32_t format, void* data )
{
	const uint32_t size = width * height * glTextureLayoutChannels(format) * glTextureTypeSize(format);

	if( size == 0 )
		return NULL;
		
	// generate texture objects
	uint32_t id = 0;
	
	GL(glEnable(GL_TEXTURE_2D));
	GL(glGenTextures(1, &id));
	GL(glBindTexture(GL_TEXTURE_2D, id));
	
	// set default texture parameters
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR));

	printf("[OpenGL]   creating %ux%u texture\n", width, height);
	
	// allocate texture
	GL_VERIFYN(glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, glTextureLayout(format), glTextureType(format), data));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	
	// allocate DMA PBO
	uint32_t dma = 0;
	
	GL(glGenBuffers(1, &dma));
	GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, dma));
	GL(glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW_ARB));
	GL(glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	
	
	mID     = id;
	mDMA    = dma;
	mWidth  = width;
	mHeight = height;
	mFormat = format;
	mSize   = size;
	return true;
}


// MapCUDA
void* glTexture::MapCUDA()
{
	if( !mInteropCUDA )
	{
		if( CUDA_FAILED(cudaGraphicsGLRegisterBuffer(&mInteropCUDA, mDMA, cudaGraphicsRegisterFlagsWriteDiscard)) )
			return NULL;

		printf( "[cuda]   registered %u byte openGL texture for interop access (%ux%u)\n", mSize, mWidth, mHeight);
	}
	
	if( CUDA_FAILED(cudaGraphicsMapResources(1, &mInteropCUDA)) )
		return NULL;
	
	void*  devPtr     = NULL;
	size_t mappedSize = 0;

	if( CUDA_FAILED(cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, mInteropCUDA)) )
	{
		CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA));
		return NULL;
	}
	
	if( mSize != mappedSize )
		printf("[OpenGL]  glTexture::MapCUDA() -- size mismatch %zu bytes  (expected=%u)\n", mappedSize, mSize);
		
	return devPtr;
}


// Unmap
void glTexture::Unmap()
{
	if( !mInteropCUDA )
		return;
		
	CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA));
	
	GL(glEnable(GL_TEXTURE_2D));
	GL(glBindTexture(GL_TEXTURE_2D, mID));
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
	
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));
}


// Upload
bool glTexture::UploadCPU( void* data )
{
	// activate texture & pbo
	GL(glEnable(GL_TEXTURE_2D));
	GL(glActiveTextureARB(GL_TEXTURE0_ARB));
	GL(glBindTexture(GL_TEXTURE_2D, mID));
	GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0));
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));

	//GL(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
	//GL(glPixelStorei(GL_UNPACK_ROW_LENGTH, img->GetWidth()));
	//GL(glPixelStorei(GL_UNPACK_IMAGE_HEIGHT, img->GetHeight()));

	// hint to driver to double-buffer
	// glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, mImage->GetSize(), NULL, GL_STREAM_DRAW_ARB);	

	// map PBO
	GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	        
	if( !ptr )
	{
		GL_CHECK("glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB)");
		return NULL;
	}

	memcpy(ptr, data, mSize);

	GL(glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB)); 

	//GL(glEnable(GL_TEXTURE_2D));
	//GL(glBindTexture(GL_TEXTURE_2D, mID));
	//GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, mDMA));
	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
	
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));

	/*if( !mInteropHost || !mInteropDevice )
	{
		if( !cudaAllocMapped(&mInteropHost, &mInteropDevice, mSize) )
			return false;
	}
	
	memcpy(mInteropHost, data, mSize);
	
	void* devGL = MapCUDA();
	
	if( !devGL )
		return false;
		
	CUDA(cudaMemcpy(devGL, mInteropDevice, mSize, cudaMemcpyDeviceToDevice));
	Unmap();*/

	return true;
}

	
// Render
void glTexture::Render( const float4& rect )
{
	GL(glEnable(GL_TEXTURE_2D));
	GL(glBindTexture(GL_TEXTURE_2D, mID));

	glBegin(GL_QUADS);

		glColor4f(1.0f,1.0f,1.0f,1.0f);

		glTexCoord2f(0.0f, 0.0f); 
		glVertex2d(rect.x, rect.y);

		glTexCoord2f(1.0f, 0.0f); 
		glVertex2d(rect.z, rect.y);	

		glTexCoord2f(1.0f, 1.0f); 
		glVertex2d(rect.z, rect.w);

		glTexCoord2f(0.0f, 1.0f); 
		glVertex2d(rect.x, rect.w);

	glEnd();

	GL(glBindTexture(GL_TEXTURE_2D, 0));
}



void glTexture::Render( float x, float y )
{
	Render(x, y, mWidth, mHeight);
}

void glTexture::Render( float x, float y, float width, float height )
{
	Render(make_float4(x, y, x + width, y + height));
}


