/*
 * inference-101
 */
 
#ifndef __GL_TEXTURE_H__
#define __GL_TEXTURE_H__

#include "cudaUtility.h"
#include "cuda_gl_interop.h"

#define USE_SDL 1
#if USE_SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_surface.h>
#include <SDL2/SDL_ttf.h>
#endif

#define MAX_IMAGES 200
/**
 * OpenGL texture
 */
class glTexture
{
public:
	static glTexture* Create( uint32_t width, uint32_t height, uint32_t format, void* data=NULL);
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
#if USE_SDL
    void Render( SDL_Renderer *renderer );
    void RenderText(char * message, SDL_Color color, int x, int y, int size);
    void Box(int x, int y, int xx, int yy);
    int ImageLoad(char * file);
    void Image(int x, int y, int id);
#endif
	
private:
	glTexture();
	bool init(uint32_t width, uint32_t height, uint32_t format, void* data);
	
	uint32_t mID;
	uint32_t mDMA;
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mFormat;
	uint32_t mSize;

#if USE_SDL
    GLuint mTextureIds[MAX_IMAGES];
    int mImageCount;
    SDL_Texture *mTextureFont;
    SDL_Rect mRect;
    SDL_Renderer *mRenderer;    
    TTF_Font *mFont18;
    TTF_Font *mFont28;
    TTF_Font *mFont36;
#endif
	
	cudaGraphicsResource* mInteropCUDA;
	void* mInteropHost;
	void* mInteropDevice;
};


#endif
