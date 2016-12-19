/*
 * camera base class
 */

#ifndef __CAMERA_H__
#define __CAMERA_H__
#include <climits>
class camera
{
public:
	camera(int height, int width) { mWidth = width; mHeight = height; };
	~camera() {};

	virtual bool Open() = 0;

	virtual void Close() = 0;
	
	// Capture frame
	virtual bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX ) = 0;
	
	inline uint32_t GetWidth() const	  { return mWidth; }
	inline uint32_t GetHeight() const	  { return mHeight; }
	inline uint32_t GetPixelDepth() const { return mDepth; }
	inline uint32_t GetSize() const		  { return mSize; }
	
protected:
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mDepth;
	uint32_t mSize;
};

#endif
