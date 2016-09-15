/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaFont.h"
#include "loadImage.h"


// constructor
cudaFont::cudaFont()
{
	mFontMapCPU = NULL;
	mFontMapGPU = NULL;
	
	mFontMapWidth  = 0;
	mFontMapHeight = 0;
}



// destructor
cudaFont::~cudaFont()
{
	if( mFontMapCPU != NULL )
	{
		CUDA(cudaFreeHost(mFontMapCPU));
		
		mFontMapCPU = NULL; 
		mFontMapGPU = NULL;
	}
}


// Create
cudaFont* cudaFont::Create( const char* bitmap_path )
{
	cudaFont* c = new cudaFont();
	
	if( !c )
		return NULL;
		
	if( !c->init(bitmap_path) )
		return NULL;
		
	return c;
}


// init
bool cudaFont::init( const char* bitmap_path )
{
	if( !loadImageRGBA(bitmap_path, &mFontMapCPU, &mFontMapGPU, &mFontMapWidth, &mFontMapHeight) )
		return false;
		
	return true;
}
