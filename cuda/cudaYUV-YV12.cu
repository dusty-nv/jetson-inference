/*
 * inference-101
 */

#include "cudaYUV.h"





inline __device__ void rgb_to_y(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y)
{
	y = static_cast<uint8_t>(((int)(30 * r) + (int)(59 * g) + (int)(11 * b)) / 100);
}

inline __device__ void rgb_to_yuv(const uint8_t r, const uint8_t g, const uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v)
{
	rgb_to_y(r, g, b, y);
	u = static_cast<uint8_t>(((int)(-17 * r) - (int)(33 * g) + (int)(50 * b) + 12800) / 100);
	v = static_cast<uint8_t>(((int)(50 * r) - (int)(42 * g) - (int)(8 * b) + 12800) / 100);
}

template <typename T, bool formatYV12>
__global__ void RGB_to_YV12( T* src, int srcAlignedWidth, uint8_t* dst, int dstPitch, int width, int height )
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	const int x1 = x + 1;
	const int y1 = y + 1;

	if( x1 >= width || y1 >= height )
		return;

	const int planeSize = height * dstPitch;
	
	uint8_t* y_plane = dst;
	uint8_t* u_plane;
	uint8_t* v_plane;

	if( formatYV12 )
	{
		u_plane = y_plane + planeSize;
		v_plane = u_plane + (planeSize / 4);	// size of U & V planes is 25% of Y plane
	}
	else
	{
		v_plane = y_plane + planeSize;		// in I420, order of U & V planes is reversed
		u_plane = v_plane + (planeSize / 4);
	}

	T px;
	uint8_t y_val, u_val, v_val;

	px = src[y * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x] = y_val;

	px = src[y * srcAlignedWidth + x1];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y * dstPitch + x1] = y_val;

	px = src[y1 * srcAlignedWidth + x];
	rgb_to_y(px.x, px.y, px.z, y_val);
	y_plane[y1 * dstPitch + x] = y_val;
	
	px = src[y1 * srcAlignedWidth + x1];
	rgb_to_yuv(px.x, px.y, px.z, y_val, u_val, v_val);
	y_plane[y1 * dstPitch + x1] = y_val;

	const int uvPitch = dstPitch / 2;
	const int uvIndex = (y / 2) * uvPitch + (x / 2);

	u_plane[uvIndex] = u_val;
	v_plane[uvIndex] = v_val;
} 

template<typename T, bool formatYV12>
cudaError_t launch420( T* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height)
{
	if( !input || !inputPitch || !output || !outputPitch || !width || !height )
		return cudaErrorInvalidValue;

	const dim3 block(32, 8);
	const dim3 grid(iDivUp(width, block.x * 2), iDivUp(height, block.y * 2));

	const int inputAlignedWidth = inputPitch / sizeof(T);

	RGB_to_YV12<T, formatYV12><<<grid, block>>>(input, inputAlignedWidth, output, outputPitch, width, height);

	return CUDA(cudaGetLastError());
}



// cudaRGBAToYV12
cudaError_t cudaRGBAToYV12( uchar4* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height )
{
	return launch420<uchar4,false>( input, inputPitch, output, outputPitch, width, height );
}

// cudaRGBAToYV12
cudaError_t cudaRGBAToYV12( uchar4* input, uint8_t* output, size_t width, size_t height )
{
	return cudaRGBAToYV12( input, width * sizeof(uchar4), output, width * sizeof(uint8_t), width, height );
}

// cudaRGBAToI420
cudaError_t cudaRGBAToI420( uchar4* input, size_t inputPitch, uint8_t* output, size_t outputPitch, size_t width, size_t height )
{
	return launch420<uchar4,true>( input, inputPitch, output, outputPitch, width, height );
}

// cudaRGBAToI420
cudaError_t cudaRGBAToI420( uchar4* input, uint8_t* output, size_t width, size_t height )
{
	return cudaRGBAToI420( input, width * sizeof(uchar4), output, width * sizeof(uint8_t), width, height );
}



#if 0
__global__ void Gray_to_YV12(const GlobPtrSz<uint8_t> src, GlobPtr<uint8_t> dst)
{
	const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
	const int y = (blockIdx.y * blockDim.y + threadIdx.y) * 2;

	if (x + 1 >= src.cols || y + 1 >= src.rows)
		return;

	// get pointers to the data
	const size_t planeSize = src.rows * dst.step;
   GlobPtr<uint8_t> y_plane = globPtr(dst.data, dst.step);
   GlobPtr<uint8_t> u_plane = globPtr(y_plane.data + planeSize, dst.step / 2);
   GlobPtr<uint8_t> v_plane = globPtr(u_plane.data + (planeSize / 4), dst.step / 2);

   uint8_t pix;
   uint8_t y_val, u_val, v_val;

   pix = src(y, x);
   rgb_to_y(pix, pix, pix, y_val);
   y_plane(y, x) = y_val;

   pix = src(y, x + 1);
   rgb_to_y(pix, pix, pix, y_val);
   y_plane(y, x + 1) = y_val;

   pix = src(y + 1, x);
   rgb_to_y(pix, pix, pix, y_val);
   y_plane(y + 1, x) = y_val;

   pix = src(y + 1, x + 1);
   rgb_to_yuv(pix, pix, pix, y_val, u_val, v_val);
   y_plane(y + 1, x + 1) = y_val;
   u_plane(y / 2, x / 2) = u_val;
   v_plane(y / 2, x / 2) = v_val;
}
#endif

