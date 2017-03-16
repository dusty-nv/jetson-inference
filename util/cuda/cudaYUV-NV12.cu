/*
 * inference-101
 */

#include "cudaYUV.h"


#define COLOR_COMPONENT_MASK            0x3FF
#define COLOR_COMPONENT_BIT_SIZE        10

#define FIXED_DECIMAL_POINT             24
#define FIXED_POINT_MULTIPLIER          1.0f
#define FIXED_COLOR_COMPONENT_MASK      0xffffffff

#define MUL(x,y)    (x*y)



__constant__ uint32_t constAlpha;
__constant__ float  constHueColorSpaceMat[9];



__device__ void YUV2RGB(uint32_t *yuvi, float *red, float *green, float *blue)
{
   

    // Prepare for hue adjustment
    /*
	 float luma, chromaCb, chromaCr;

	luma     = (float)yuvi[0];
    chromaCb = (float)((int)yuvi[1] - 512.0f);
    chromaCr = (float)((int)yuvi[2] - 512.0f);

    // Convert YUV To RGB with hue adjustment
    *red  = MUL(luma,     constHueColorSpaceMat[0]) +
            MUL(chromaCb, constHueColorSpaceMat[1]) +
            MUL(chromaCr, constHueColorSpaceMat[2]);
    *green= MUL(luma,     constHueColorSpaceMat[3]) +
            MUL(chromaCb, constHueColorSpaceMat[4]) +
            MUL(chromaCr, constHueColorSpaceMat[5]);
    *blue = MUL(luma,     constHueColorSpaceMat[6]) +
            MUL(chromaCb, constHueColorSpaceMat[7]) +
            MUL(chromaCr, constHueColorSpaceMat[8]);*/

	const float luma = float(yuvi[0]);
	const float u    = float(yuvi[1]) - 512.0f;
	const float v    = float(yuvi[2]) - 512.0f;

   /*R = Y + 1.140V
   G = Y - 0.395U - 0.581V
   B = Y + 2.032U*/

	/**green = luma + 1.140f * v;
	*blue  = luma - 0.395f * u - 0.581f * v;
	*red   = luma + 2.032f * u;*/

	*red    = luma + 1.140f * v;
	*green  = luma - 0.395f * u - 0.581f * v;
	*blue   = luma + 2.032f * u;
}


__device__ uint32_t RGBAPACK_8bit(float red, float green, float blue, uint32_t alpha)
{
    uint32_t ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 255.0f);
    green = min(max(green, 0.0f), 255.0f);
    blue  = min(max(blue,  0.0f), 255.0f);

    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = ((((uint32_t)red)   << 24) |
                 (((uint32_t)green) << 16) |
		       (((uint32_t)blue)  <<  8) | (uint32_t)alpha);

    return  ARGBpixel;
}


__device__ uint32_t RGBAPACK_10bit(float red, float green, float blue, uint32_t alpha)
{
    uint32_t ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 1023.f);
    green = min(max(green, 0.0f), 1023.f);
    blue  = min(max(blue,  0.0f), 1023.f);

    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = ((((uint32_t)red   >> 2) << 24) |
                 (((uint32_t)green >> 2) << 16) |
                 (((uint32_t)blue  >> 2) <<  8) | (uint32_t)alpha);

    return  ARGBpixel;
}


// CUDA kernel for outputing the final ARGB output from NV12;
/*extern "C"*/
__global__ void Passthru(uint32_t *srcImage,   size_t nSourcePitch,
                         uint32_t *dstImage,   size_t nDestPitch,
                         uint32_t width,       uint32_t height)
{
    int x, y;
    uint32_t yuv101010Pel[2];
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint32_t dstImagePitch   = nDestPitch >> 2;
    uint8_t *srcImageU8     = (uint8_t *)srcImage;

    processingPitch = nSourcePitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]);
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]);

    // this steps performs the color conversion
    float luma[2];

    luma[0]   = (yuv101010Pel[0]        & 0x00FF);
    luma[1]   = (yuv101010Pel[1]        & 0x00FF);

    // Clamp the results to RGBA
    dstImage[y * dstImagePitch + x     ] = RGBAPACK_8bit(luma[0], luma[0], luma[0], constAlpha);
    dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_8bit(luma[1], luma[1], luma[1], constAlpha);
}


// CUDA kernel for outputing the final ARGB output from NV12;
/*extern "C"*/
__global__ void NV12ToARGB(uint32_t *srcImage,     size_t nSourcePitch,
                           uint32_t *dstImage,     size_t nDestPitch,
                           uint32_t width,         uint32_t height)
{
    int x, y;
    uint32_t yuv101010Pel[2];
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint32_t dstImagePitch   = nDestPitch >> 2;
    uint8_t *srcImageU8     = (uint8_t *)srcImage;

    processingPitch = nSourcePitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]) << 2;
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

    uint32_t chromaOffset    = processingPitch * height;
    int y_chroma = y >> 1;

    if (y & 1)  // odd scanline ?
    {
        uint32_t chromaCb;
        uint32_t chromaCr;

        chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x    ];
        chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

        if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
        {
            chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x    ] + 1) >> 1;
            chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
        }

        yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }
    else
    {
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }

    // this steps performs the color conversion
    uint32_t yuvi[6];
    float red[2], green[2], blue[2];

    yuvi[0] = (yuv101010Pel[0] &   COLOR_COMPONENT_MASK);
    yuvi[1] = ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    yuvi[3] = (yuv101010Pel[1] &   COLOR_COMPONENT_MASK);
    yuvi[4] = ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
    YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

    // Clamp the results to RGBA
    dstImage[y * dstImagePitch + x     ] = RGBAPACK_10bit(red[0], green[0], blue[0], constAlpha);
    dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_10bit(red[1], green[1], blue[1], constAlpha);
}


bool nv12ColorspaceSetup = false;


// cudaNV12ToARGB32
cudaError_t cudaNV12ToRGBA( uint8_t* srcDev, size_t srcPitch, uchar4* destDev, size_t destPitch, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	if( !nv12ColorspaceSetup )
		cudaNV12SetupColorspace();

	const dim3 blockDim(32,16,1);
	const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);

	NV12ToARGB<<<gridDim, blockDim>>>( (uint32_t*)srcDev, srcPitch, (uint32_t*)destDev, destPitch, width, height );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaNV12ToRGBA( uint8_t* srcDev, uchar4* destDev, size_t width, size_t height )
{
	return cudaNV12ToRGBA(srcDev, width * sizeof(uint8_t), destDev, width * sizeof(uchar4), width, height);
}


//-------------------------------------------------------------------------------------------------------------------------

__global__ void NV12ToRGBAf(uint32_t* srcImage,  size_t nSourcePitch,
                           float4* dstImage,     size_t nDestPitch,
                           uint32_t width,       uint32_t height)
{
    int x, y;
    uint32_t yuv101010Pel[2];
    uint32_t processingPitch = ((width) + 63) & ~63;
    uint8_t *srcImageU8     = (uint8_t *)srcImage;

    processingPitch = nSourcePitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

#if 1	
    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]) << 2;
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

    uint32_t chromaOffset    = processingPitch * height;
    int y_chroma = y >> 1;

    if (y & 1)  // odd scanline ?
    {
        uint32_t chromaCb;
        uint32_t chromaCr;

        chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x    ];
        chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

        if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
        {
            chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x    ] + 1) >> 1;
            chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
        }

        yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }
    else
    {
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= ((uint32_t)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }

    // this steps performs the color conversion
    uint32_t yuvi[6];
    float red[2], green[2], blue[2];

    yuvi[0] = (yuv101010Pel[0] &   COLOR_COMPONENT_MASK);
    yuvi[1] = ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    yuvi[3] = (yuv101010Pel[1] &   COLOR_COMPONENT_MASK);
    yuvi[4] = ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
    YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

    // Clamp the results to RGBA
	//printf("cuda thread %i %i  %f %f %f\n", x, y, red[0], green[0], blue[0]);

	const float s = 1.0f / 1024.0f * 255.0f;

	dstImage[y * width + x]     = make_float4(red[0] * s, green[0] * s, blue[0] * s, 1.0f);
	dstImage[y * width + x + 1] = make_float4(red[1] * s, green[1] * s, blue[1] * s, 1.0f);
#else
	//printf("cuda thread %i %i  %i %i \n", x, y, width, height);
		
	dstImage[y * width + x]     = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
	dstImage[y * width + x + 1] = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
#endif
}



// cudaNV12ToRGBA
cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, size_t srcPitch, float4* destDev, size_t destPitch, size_t width, size_t height )
{
	if( !srcDev || !destDev )
		return cudaErrorInvalidDevicePointer;

	if( srcPitch == 0 || destPitch == 0 || width == 0 || height == 0 )
		return cudaErrorInvalidValue;

	if( !nv12ColorspaceSetup )
		cudaNV12SetupColorspace();

	const dim3 blockDim(8,8,1);
	//const dim3 gridDim((width+(2*blockDim.x-1))/(2*blockDim.x), (height+(blockDim.y-1))/blockDim.y, 1);
	const dim3 gridDim(iDivUp(width,blockDim.x), iDivUp(height, blockDim.y), 1);

	NV12ToRGBAf<<<gridDim, blockDim>>>( (uint32_t*)srcDev, srcPitch, destDev, destPitch, width, height );
	
	return CUDA(cudaGetLastError());
}

cudaError_t cudaNV12ToRGBAf( uint8_t* srcDev, float4* destDev, size_t width, size_t height )
{
	return cudaNV12ToRGBAf(srcDev, width * sizeof(uint8_t), destDev, width * sizeof(float4), width, height);
}


// cudaNV12SetupColorspace
cudaError_t cudaNV12SetupColorspace( float hue )
{
	const float hueSin = sin(hue);
	const float hueCos = cos(hue);

	float hueCSC[9];

	const bool itu601 = false;

	if( itu601 /*CSC == ITU601*/)
	{
		//CCIR 601
		hueCSC[0] = 1.1644f;
		hueCSC[1] = hueSin * 1.5960f;
		hueCSC[2] = hueCos * 1.5960f;
		hueCSC[3] = 1.1644f;
		hueCSC[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
		hueCSC[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
		hueCSC[6] = 1.1644f;
		hueCSC[7] = hueCos *  2.0172f;
		hueCSC[8] = hueSin * -2.0172f;
	}
	else /*if(CSC == ITU709)*/
	{
		//CCIR 709
		hueCSC[0] = 1.0f;
		hueCSC[1] = hueSin * 1.57480f;
		hueCSC[2] = hueCos * 1.57480f;
		hueCSC[3] = 1.0;
		hueCSC[4] = (hueCos * -0.18732f) - (hueSin * 0.46812f);
		hueCSC[5] = (hueSin *  0.18732f) - (hueCos * 0.46812f);
		hueCSC[6] = 1.0f;
		hueCSC[7] = hueCos *  1.85560f;
		hueCSC[8] = hueSin * -1.85560f;
	}


	if( CUDA_FAILED(cudaMemcpyToSymbol(constHueColorSpaceMat, hueCSC, sizeof(float) * 9)) )
		return cudaErrorInvalidSymbol;

	uint32_t cudaAlpha = ((uint32_t)0xff<< 24);

	if( CUDA_FAILED(cudaMemcpyToSymbol(constAlpha, &cudaAlpha, sizeof(uint32_t))) )
		return cudaErrorInvalidSymbol;

	nv12ColorspaceSetup = true;
	return cudaSuccess;
}

