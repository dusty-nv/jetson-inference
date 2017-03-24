/*
 * inference-101
 */

#ifndef __V4L2_CAPTURE_H
#define __V4L2_CAPTURE_H


#include <linux/videodev2.h>

#include <stdint.h>
#include <string>
#include <vector>



struct v4l2_mmap
{
	struct v4l2_buffer buf;
	void*  ptr;
};


/**
 * Video4Linux2 camera capture streaming.
 * @ingroup util
 */
class v4l2Camera
{
public:	
	/**
	 * Create V4L2 interface
	 * @param path Filename of the video device (e.g. /dev/video0)
	 */
	static v4l2Camera* Create( const char* device_path );

	/**
	 * Destructor
	 */	
	~v4l2Camera();

	/**
 	 * Start streaming
	 */
	bool Open();

	/**
	 * Stop streaming
	 */
	bool Close();

	/**
	 * Return the next image.
	 */
	void* Capture( size_t timeout=0 );

	/**
	 * Get width, in pixels, of camera image.
	 */
	inline uint32_t GetWidth() const					{ return mWidth; }
	
	/**
	 * Retrieve height, in pixels, of camera image.
	 */
	inline uint32_t GetHeight() const					{ return mHeight; }

	/**
 	 * Return the size in bytes of one line of the image.
	 */
	inline uint32_t GetPitch() const					{ return mPitch; }

	/**
	 * Return the bit depth per pixel.
	 */
	inline uint32_t GetPixelDepth() const				{ return mPixelDepth; }

private:

	v4l2Camera( const char* device_path );

	bool init();
	bool initCaps();
	bool initFormats();
	bool initStream();

	bool initUserPtr();
	bool initMMap();

	int 	mFD;
	int	    mRequestFormat;
	uint32_t mRequestWidth;
	uint32_t mRequestHeight;
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mPitch;
	uint32_t mPixelDepth;

	v4l2_mmap* mBuffersMMap;
	size_t mBufferCountMMap;

	std::vector<v4l2_fmtdesc> mFormats;
	std::string mDevicePath;
};


#endif


