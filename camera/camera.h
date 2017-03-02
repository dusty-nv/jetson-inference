/*
 * camera base class
 */

#ifndef __CAMERA_H__
#define __CAMERA_H__
#include <climits>
#include <stdint.h>

class camera
{
public:
	camera(int height, int width) { mWidth = width; mHeight = height; mRGBA = 0; };
	~camera() {};

	virtual bool Open() = 0;

	virtual void Close() = 0;
	
	// Capture frame
	virtual bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX ) = 0;
	
	inline uint32_t GetWidth() const	  { return mWidth; }
	inline uint32_t GetHeight() const	  { return mHeight; }
	inline uint32_t GetPixelDepth() const { return mDepth; }
	inline uint32_t GetSize() const		  { return mSize; }
	
	// Takes in captured YUV-NV12 CUDA image, converts to float4 RGBA (with pixel intensity 0-255)
	bool ConvertBAYER_GR8toRGBA( void* input, void** output );
	bool ConvertNV12toRGBA( void* input, void** output );
	bool ConvertYUVtoRGBA ( void* input, void** output );
	bool ConvertRGBtoRGBA ( void* input, void** output );
	bool ConvertYUVtoRGBf ( void* input, void** output );
	
protected:
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mDepth;
	uint32_t mSize;
	
	void* mRGBA;
};

#endif
