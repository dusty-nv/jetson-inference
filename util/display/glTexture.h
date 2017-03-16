/*
 * inference-101
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
