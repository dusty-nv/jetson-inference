/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "cudaYUV.h"


inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}


/* From RGB to YUV

   Y = 0.299R + 0.587G + 0.114B
   U = 0.492 (B-Y)
   V = 0.877 (R-Y)

   It can also be represented as:

   Y =  0.299R + 0.587G + 0.114B
   U = -0.147R - 0.289G + 0.436B
   V =  0.615R - 0.515G - 0.100B

   From YUV to RGB

   R = Y + 1.140V
   G = Y - 0.395U - 0.581V
   B = Y + 2.032U
 */

struct __align__(8) uchar8
{
   uint8_t a0, a1, a2, a3, a4, a5, a6, a7;
};
static __host__ __device__ __forceinline__ uchar8 make_uchar8(uint8_t a0, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7)
{
   uchar8 val = {a0, a1, a2, a3, a4, a5, a6, a7};
   return val;
}


//-----------------------------------------------------------------------------------
// YUYV/UYVY to RGBA
//-----------------------------------------------------------------------------------
template <bool formatUYVY>
__global__ void yuyvToRgba( uchar4* src, int srcAlignedWidth, uchar8* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	// Y0 is the brightness of pixel 0, Y1 the brightness of pixel 1.
	// U0 and V0 is the color of both pixels.
	// UYVY [ U0 | Y0 | V0 | Y1 ] 
	// YUYV [ Y0 | U0 | Y1 | V0 ]
	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 
	const float u = (formatUYVY ? macroPx.x : macroPx.y) - 128.0f;
	const float v = (formatUYVY ? macroPx.z : macroPx.w) - 128.0f;

	const float4 px0 = make_float4( y0 + 1.4065f * v,
							  y0 - 0.3455f * u - 0.7169f * v,
							  y0 + 1.7790f * u, 255.0f );

	const float4 px1 = make_float4( y1 + 1.4065f * v,
							  y1 - 0.3455f * u - 0.7169f * v,
							  y1 + 1.7790f * u, 255.0f );

	dst[y * dstAlignedWidth + x] = make_uchar8( clamp(px0.x, 0.0f, 255.0f), 
									    clamp(px0.y, 0.0f, 255.0f),
									    clamp(px0.z, 0.0f, 255.0f),
									    clamp(px0.w, 0.0f, 255.0f),
									    clamp(px1.x, 0.0f, 255.0f),
									    clamp(px1.y, 0.0f, 255.0f),
									    clamp(px1.z, 0.0f, 255.0f),
									    clamp(px1.w, 0.0f, 255.0f) );
} 

template<bool formatUYVY>
cudaError_t launchYUYV( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(uchar8);	// normally would be uchar4 ^^^

	//printf("yuyvToRgba %zu %zu %i %i %i %i %i\n", width, height, (int)formatUYVY, srcAlignedWidth, dstAlignedWidth, grid.x, grid.y);

	yuyvToRgba<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (uchar8*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}


cudaError_t cudaUYVYToRGBA( uchar2* input, uchar4* output, size_t width, size_t height )
{
	return cudaUYVYToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

cudaError_t cudaUYVYToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToRGBA( uchar2* input, uchar4* output, size_t width, size_t height )
{
	return cudaYUYVToRGBA(input, width * sizeof(uchar2), output, width * sizeof(uchar4), width, height);
}

cudaError_t cudaYUYVToRGBA( uchar2* input, size_t inputPitch, uchar4* output, size_t outputPitch, size_t width, size_t height )
{
	return launchYUYV<false>(input, inputPitch, output, outputPitch, width, height);
}


//-----------------------------------------------------------------------------------
// YUYV/UYVY to grayscale
//-----------------------------------------------------------------------------------

template <bool formatUYVY>
__global__ void yuyvToGray( uchar4* src, int srcAlignedWidth, float2* dst, int dstAlignedWidth, int width, int height )
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x >= srcAlignedWidth || y >= height )
		return;

	const uchar4 macroPx = src[y * srcAlignedWidth + x];

	const float y0 = formatUYVY ? macroPx.y : macroPx.x;
	const float y1 = formatUYVY ? macroPx.w : macroPx.z; 

	dst[y * dstAlignedWidth + x] = make_float2(y0/255.0f, y1/255.0f);
} 

template<bool formatUYVY>
cudaError_t launchGrayYUYV( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(8,8);
	const dim3 grid(iDivUp(width/2, block.x), iDivUp(height, block.y));

	const int srcAlignedWidth = inputPitch / sizeof(uchar4);	// normally would be uchar2, but we're doubling up pixels
	const int dstAlignedWidth = outputPitch / sizeof(float2);	// normally would be float ^^^

	yuyvToGray<formatUYVY><<<grid, block>>>((uchar4*)input, srcAlignedWidth, (float2*)output, dstAlignedWidth, width, height);

	return CUDA(cudaGetLastError());
}

cudaError_t cudaUYVYToGray( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaUYVYToGray(input, width * sizeof(uchar2), output, width * sizeof(uint8_t), width, height);
}

cudaError_t cudaUYVYToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<true>(input, inputPitch, output, outputPitch, width, height);
}

cudaError_t cudaYUYVToGray( uchar2* input, float* output, size_t width, size_t height )
{
	return cudaYUYVToGray(input, width * sizeof(uchar2), output, width * sizeof(float), width, height);
}

cudaError_t cudaYUYVToGray( uchar2* input, size_t inputPitch, float* output, size_t outputPitch, size_t width, size_t height )
{
	return launchGrayYUYV<false>(input, inputPitch, output, outputPitch, width, height);
}

