/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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


