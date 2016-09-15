/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaFont.h"
#include "cudaMappedMemory.h"

#include "loadImage.h"


// constructor
cudaFont::cudaFont()
{
	mCommandCPU = NULL;
	mCommandGPU = NULL;
	mCmdEntries = 0;

	mFontMapCPU = NULL;
	mFontMapGPU = NULL;
	
	mFontMapWidth  = 0;
	mFontMapHeight = 0;
	
	mFontCellSize = make_int2(24,32);
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
	
	if( !cudaAllocMapped((void**)&mCommandCPU, (void**)&mCommandGPU, sizeof(short4) * MaxCommands) )
		return false;
		
	return true;
}


template<typename T>
__global__ void gpuOverlayText( T* font, int fontWidth, short4* text,
						        T* output, int width, int height ) 
{
	const short4 t = text[blockIdx.x];

	//printf("%i %hi %hi %hi %hi\n", blockIdx.x, t.x, t.y, t.z, t.w);

	const int x = t.x + threadIdx.x;
	const int y = t.y + threadIdx.y;

	if( x < 0 || y < 0 || x >= width || y >= height )
		return;

	const int u = t.z + threadIdx.x;
	const int v = t.w + threadIdx.y;

	//printf("%i %i %i %i %i\n", blockIdx.x, x, y, u, v);

	output[y * width + x] = font[v * fontWidth + u];	 
}


// processCUDA
template<typename T>
cudaError_t cudaOverlayText( T* font, const int2& fontCellSize, size_t fontMapWidth,
					    short4* text, size_t length,
					    T* output, size_t width, size_t height)	
{
	if( !font || !text || !output || length == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	// setup arguments
	const dim3 block(fontCellSize.x, fontCellSize.y);
	const dim3 grid(length);

	gpuOverlayText<<<grid, block>>>(font, fontMapWidth, text, output, width, height); 

	return cudaGetLastError();
}


// RenderOverlay
bool cudaFont::RenderOverlay( float4* input, float4* output, uint32_t width, uint32_t height, const std::vector< std::pair< std::string, int2 > >& text )
{
	if( !input || !output || width == 0 || height == 0 || text.size() == 0 )
		return false;
	
	const uint32_t cellsPerRow = mFontMapWidth / mFontCellSize.x;
	const uint32_t numText     = text.size();
	
	for( uint32_t t=0; t < numText; t++ )
	{
		const uint32_t numChars = text[t].first.size();
		
		int2 pos = text[t].second;
		
		for( uint32_t n=0; n < numChars; n++ )
		{
			char c = text[t].first[n];
			
			if( c < 32 || c > 126 )
				continue;
			
			c -= 32;
			
			const uint32_t font_y = c / cellsPerRow;
			const uint32_t font_x = c - (font_y * cellsPerRow);
			
			mCommandCPU[mCmdEntries++] = make_short4( pos.x, pos.y,
													  font_x * (mFontCellSize.x + 1),
													  font_y * (mFontCellSize.y + 1) );
		
			pos.x += mFontCellSize.x;
		}
	}

	CUDA(cudaOverlayText<float4>( mFontMapGPU, mFontCellSize, mFontMapWidth,
				        mCommandGPU, mCmdEntries, 
				       output, width, height));
					   
	mCmdEntries = 0;
	return true;
}

	
