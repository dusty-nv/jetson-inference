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


inline __host__ __device__ float4 operator*(float4 a, float4 b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template<typename T>
__global__ void gpuOverlayText( T* font, int fontWidth, short4* text,
						        T* output, int width, int height, float4 color ) 
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
	
	const T px_font = font[v * fontWidth + u] * color;
	      T px_out  = output[y * width + x];	// fixme:  add proper input support

	const float alpha = px_font.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	px_out.x = alpha * px_font.x + ialph * px_out.x;
	px_out.y = alpha * px_font.y + ialph * px_out.y;
	px_out.z = alpha * px_font.z + ialph * px_out.z; 

	output[y * width + x] = px_out;	 
}


// processCUDA
template<typename T>
cudaError_t cudaOverlayText( T* font, const int2& fontCellSize, size_t fontMapWidth,
					    const float4& fontColor, short4* text, size_t length,
					    T* output, size_t width, size_t height)	
{
	if( !font || !text || !output || length == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	const float4 color_scale = make_float4( fontColor.x / 255.0f, fontColor.y / 255.0f, fontColor.z / 255.0f, fontColor.w / 255.0f );
	
	// setup arguments
	const dim3 block(fontCellSize.x, fontCellSize.y);
	const dim3 grid(length);

	gpuOverlayText<<<grid, block>>>(font, fontMapWidth, text, output, width, height, color_scale); 

	return cudaGetLastError();
}


// RenderOverlay
bool cudaFont::RenderOverlay( float4* input, float4* output, uint32_t width, uint32_t height, const std::vector< std::pair< std::string, int2 > >& text, const float4& color )
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

	CUDA(cudaOverlayText<float4>( mFontMapGPU, mFontCellSize, mFontMapWidth, color,
				        mCommandGPU, mCmdEntries, 
				       output, width, height));
					   
	mCmdEntries = 0;
	return true;
}


bool cudaFont::RenderOverlay( float4* input, float4* output, uint32_t width, uint32_t height, 
							  const char* str, int x, int y, const float4& color )
{
	if( !str )
		return NULL;
		
	std::vector< std::pair< std::string, int2 > > list;
	
	list.push_back( std::pair< std::string, int2 >( str, make_int2(x,y) ));
	
	return RenderOverlay(input, output, width, height, list, color);
}
						
	
