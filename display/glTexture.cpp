/*
 * inference-101
 */

#include "glUtility.h"
#include "glTexture.h"
#include "cudaMappedMemory.h"

#define BIN_ROOT "../../../data/"
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
    mImageCount = 0;

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
glTexture* glTexture::Create( uint32_t width, uint32_t height, uint32_t format, void* data)
{
	glTexture* tex = new glTexture();
	
	if( !tex->init(width, height, format, data) )
	{
		printf("[OpenGL]  failed to create %ux%u texture\n", width, height);
		return NULL;
	}
	
	return tex;
}

void glTexture::Render( SDL_Renderer *renderer ) 
{
    mRenderer = renderer; 
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

	debug_print("[OpenGL] creating %ux%u texture\n", width, height);
	
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
// printf(">> %s:%d\n",__FUNCTION__,__LINE__);

#if USE_SDL
    SDL_Init( SDL_INIT_EVERYTHING );
    if (TTF_Init()==-1){
        fprintf(stderr, "error: TTF init error\n");
        exit(EXIT_FAILURE);
    }

    mFont18 = TTF_OpenFont(BIN_ROOT "Roboto-Regular.ttf", 18);
    mFont28 = TTF_OpenFont(BIN_ROOT "Roboto-Regular.ttf", 28);
    mFont36 = TTF_OpenFont(BIN_ROOT "Roboto-Regular.ttf", 36);
    if (mFont28 == NULL) {
        fprintf(stderr, "error: font not found\n");
        exit(EXIT_FAILURE);
    }
#endif
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

	// map PBO
	GLubyte* ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
	        
	if( !ptr )
	{
		GL_CHECK("glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB)");
		return NULL;
	}

	memcpy(ptr, data, mSize);

	GL(glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB)); 

	GL(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, mWidth, mHeight, glTextureLayout(mFormat), glTextureType(mFormat), NULL));
	
	GL(glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0));
	GL(glBindTexture(GL_TEXTURE_2D, 0));
	GL(glDisable(GL_TEXTURE_2D));

	return true;
}

#if USE_SDL
/*
  // Prints out "Hello World" at location (5,10) at font size 12!
  SDL_Color color = {255, 0, 0, 0}; // Red
  RenderText("Hello World", color, 5, 10, 12); 
*/

void glTexture::RenderText(char * message, SDL_Color color, int x, int y, int size) 
{
  SDL_Surface * sFont;
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_TEXTURE_2D);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);

  switch (size)
  {
  case 18:
      sFont = TTF_RenderText_Blended(mFont18, message, color);
      break;
  case 28:
      sFont = TTF_RenderText_Blended(mFont28, message, color);
      break;
  case 36:
  default:
      sFont = TTF_RenderText_Blended(mFont36, message, color);
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, sFont->w, sFont->h, 0, GL_BGRA, GL_UNSIGNED_BYTE, sFont->pixels);

  glBegin(GL_QUADS);
  {
    glColor3f(color.r*2.55, color.g*2.55, color.b*2.55);
    glTexCoord2f(0,0); glVertex2f(x, y);
    glTexCoord2f(1,0); glVertex2f(x + sFont->w, y);
    glTexCoord2f(1,1); glVertex2f(x + sFont->w, y + sFont->h);
    glTexCoord2f(0,1); glVertex2f(x, y + sFont->h);
  }
  glEnd();

  glDisable(GL_BLEND);
  glDisable(GL_TEXTURE_2D);
  glEnable(GL_DEPTH_TEST);

  glDeleteTextures(1, &texture);
  SDL_FreeSurface(sFont);
}

int glTexture::ImageLoad(char * file)
{
    int status = false;
    float h,w;
    char filename[300];
 	// generate texture objects

    /* Create storage space for the texture */
    SDL_Surface *TextureImage; 
    glEnable(GL_TEXTURE_2D);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    sprintf(filename, "%s%s", BIN_ROOT, file);

    /* Load The Bitmap, Check For Errors, If Bitmap's Not Found Quit */
    if ( ( TextureImage = SDL_LoadBMP( filename ) ) )
    {
        printf("Loaded texture abaco.bmp\n");

	    /* Set the status to true */
	    status = true;

	    /* Create The Texture */
        glGenTextures(1, &mTextureIds[mImageCount]);

	    /* Typical Texture Generation Using Data From The Bitmap */
        glBindTexture(GL_TEXTURE_2D, mTextureIds[mImageCount]);

	    /* Generate The Texture */
	    glTexImage2D( GL_TEXTURE_2D, 0, 3, TextureImage->w, TextureImage->h, 0, GL_BGR, GL_UNSIGNED_BYTE, TextureImage->pixels );

	    /* Linear Filtering */
	    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    } else
        printf("Unable to open BMP file\n");
 
    /* Free up any memory we may have used */
    if ( TextureImage )
	    SDL_FreeSurface( TextureImage );

    glDisable(GL_BLEND);
    
    return mImageCount++;
}
#endif

void glTexture::Box(int x, int y, int xx, int yy)
{	
	glEnable(GL_TEXTURE_2D);
     
    // grey box behnd text
    glBegin(GL_QUADS); 
        glColor3f(.96, .41, .13);
	    glVertex2d(x, y);
	    glVertex2d(xx, y);	
	    glVertex2d(xx, yy);
	    glVertex2d(x, yy);
    glEnd(); 
}

#if USE_SDL 
void glTexture::Image(int x, int y, int id)
// Render logo in bottom Right corner
{ 
    float h,w;
#if 1
    int offset=50;
    int width=87;  // Todo : Auto detect image size
    int height=40;

    // Draw logo
	glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, mTextureIds[id]);

#if 1
    glBegin(GL_QUADS); 
	    glTexCoord2f(0.0f, 0.0f); 
	    glVertex2d((mWidth / 2) - (width/2), mHeight - height);

	    glTexCoord2f(1.0f, 0.0f); 
	    glVertex2d((mWidth / 2) + (width/2), mHeight - height);	

	    glTexCoord2f(1.0f, 1.0f); 
	    glVertex2d((mWidth / 2) + (width/2), mHeight);

	    glTexCoord2f(0.0f, 1.0f); 
	    glVertex2d((mWidth / 2) - (width/2), mHeight);
    glEnd(); 
#endif
    glBindTexture(GL_TEXTURE_2D, 0);
}
#endif

// Render
void glTexture::Render( const float4& rect )
{
    float h,w;

	GL(glEnable(GL_TEXTURE_2D));
    glBindTexture(GL_TEXTURE_2D, mID);
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
	glBindTexture(GL_TEXTURE_2D, 0);
#endif
}

void glTexture::Render( float x, float y )
{
	Render(x, y, mWidth, mHeight);
}

void glTexture::Render( float x, float y, float width, float height )
{
	Render(make_float4(x, y, x + width, y + height));
}


