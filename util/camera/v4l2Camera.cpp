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

#include "v4l2Camera.h"

#include <fcntl.h> 
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>



#define REQUESTED_RINGBUFFERS 	4



// ioctl
static int xioctl(int fd, int request, void* arg)
{
    int status;
    do { status = ioctl (fd, request, arg); } while (-1==status && EINTR==errno);
    return status;
}



// constructor
v4l2Camera::v4l2Camera( const char* device_path ) : mDevicePath(device_path)
{	
	mFD = -1;

	mBuffersMMap     = NULL;
	mBufferCountMMap = 0;
	mRequestWidth    = 0;
	mRequestHeight   = 0;
	mRequestFormat   = 1;
	//mRequestFormat   = -1;	// index into V4L2 format table
	
	mWidth      = 0;
	mHeight     = 0;
	mPitch      = 0;
	mPixelDepth = 0;
}


// destructor	
v4l2Camera::~v4l2Camera()
{
	// close file
	if( mFD >= 0 )
	{
		close(mFD);
		mFD = -1;
	}
}


// ProcessEmit
void* v4l2Camera::Capture( size_t timeout )
{
	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(mFD, &fds);

	struct timeval tv;
 
	tv.tv_sec  = 0;
	tv.tv_usec = 0;

	const bool threaded = true; //false;

	if( timeout > 0 )
	{
		tv.tv_sec  = timeout / 1000;
		tv.tv_usec = (timeout - (tv.tv_sec * 1000)) * 1000;
	}
	
	//
	const int result = select(mFD + 1, &fds, NULL, NULL, &tv);


	if( result == -1 ) 
	{
		//if (EINTR == errno)
		printf("v4l2 -- select() failed (errno=%i) (%s)\n", errno, strerror(errno));
		return NULL;
	}
	else if( result == 0 )
	{
		if( timeout > 0 )
			printf("v4l2 -- select() timed out...\n");
		return NULL;	// timeout, not necessarily an error (TRY_AGAIN)
	}

	// dequeue input buffer from V4L2
	struct v4l2_buffer buf;
	memset(&buf, 0, sizeof(v4l2_buffer));

	buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_MMAP;	//V4L2_MEMORY_USERPTR;

	if( xioctl(mFD, VIDIOC_DQBUF, &buf) < 0 )
	{
		printf("v4l2 -- ioctl(VIDIOC_DQBUF) failed (errno=%i) (%s)\n", errno, strerror(errno));
		return NULL;
	}
	
	if( buf.index >= mBufferCountMMap )
	{
		printf("v4l2 -- invalid mmap buffer index (%u)\n", buf.index);
		return NULL;
	}
	
	// emit ringbuffer entry
	//printf("v4l2 -- recieved %ux%u video frame (index=%u)\n", mWidth, mHeight, (uint32_t)buf.index);

	void* image_ptr = mBuffersMMap[buf.index].ptr;

	// re-queue buffer to V4L2
	if( xioctl(mFD, VIDIOC_QBUF, &buf) < 0 )
		printf("v4l2 -- ioctl(VIDIOC_QBUF) failed (errno=%i) (%s)\n", errno, strerror(errno));

	return image_ptr;
}



// initMMap
bool v4l2Camera::initMMap()
{
	struct v4l2_requestbuffers req;
	memset(&req, 0, sizeof(v4l2_requestbuffers));

	req.count  = REQUESTED_RINGBUFFERS;
	req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if( xioctl(mFD, VIDIOC_REQBUFS, &req) < 0 )
	{
		printf("v4l2 -- does not support mmap (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	if( req.count < 2 )
	{
		printf("v4l2 -- insufficient mmap memory\n");
		return false;
	}

	mBuffersMMap = (v4l2_mmap*)malloc( req.count * sizeof(v4l2_mmap) );
	
	if( !mBuffersMMap )
		return false;

	memset(mBuffersMMap, 0, req.count * sizeof(v4l2_mmap));

	for( size_t n=0; n < req.count; n++ )
	{
		mBuffersMMap[n].buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		mBuffersMMap[n].buf.memory = V4L2_MEMORY_MMAP;
		mBuffersMMap[n].buf.index  = n;
		
		if( xioctl(mFD, VIDIOC_QUERYBUF, &mBuffersMMap[n].buf) < 0 )
		{
			printf( "v4l2 -- failed retrieve mmap buffer info (errno=%i) (%s)\n", errno, strerror(errno));
			return false;
		}

		mBuffersMMap[n].ptr = mmap(NULL, mBuffersMMap[n].buf.length,
							  PROT_READ|PROT_WRITE, MAP_SHARED,
							  mFD, mBuffersMMap[n].buf.m.offset);

		if( mBuffersMMap[n].ptr == MAP_FAILED )
		{
			printf( "v4l2 -- failed to mmap buffer (errno=%i) (%s)\n", errno, strerror(errno));
			return false;
		}

		if( xioctl(mFD, VIDIOC_QBUF, &mBuffersMMap[n].buf) < 0 )
		{
			printf( "v4l2 -- failed to queue mmap buffer (errno=%i) (%s)\n", errno, strerror(errno));
			return false;
		}
	}

	mBufferCountMMap = req.count;	
	printf("v4l2 -- mapped %zu capture buffers with mmap\n", mBufferCountMMap); 	
	return true;
}


inline const char* v4l2_format_str( uint32_t fmt )
{
	if( fmt == V4L2_PIX_FMT_SBGGR8 )	   return "SBGGR8 (V4L2_PIX_FMT_SBGGR8)";
	else if( fmt == V4L2_PIX_FMT_SGBRG8 )  return "SGBRG8 (V4L2_PIX_FMT_SGBRG8)";
	else if( fmt == V4L2_PIX_FMT_SGRBG8 )  return "SGRBG8 (V4L2_PIX_FMT_SGRBG8)";
	else if( fmt == V4L2_PIX_FMT_SRGGB8 )  return "SRGGB8 (V4L2_PIX_FMT_SRGGB8)";
	else if( fmt == V4L2_PIX_FMT_SBGGR16 ) return "BYR2 (V4L2_PIX_FMT_SBGGR16)";
	else if( fmt == V4L2_PIX_FMT_SRGGB10 ) return "RG10 (V4L2_PIX_FMT_SRGGB10)";
	
	return "UNKNOWN";
}


inline void v4l2_print_format( const v4l2_format& fmt, const char* text )
{
	printf("v4l2 -- %s\n", text);
	printf("v4l2 --   width  %u\n", fmt.fmt.pix.width);
	printf("v4l2 --   height %u\n", fmt.fmt.pix.height);
	printf("v4l2 --   pitch  %u\n", fmt.fmt.pix.bytesperline);
	printf("v4l2 --   size   %u\n", fmt.fmt.pix.sizeimage);
	printf("v4l2 --   format 0x%X  %s\n", fmt.fmt.pix.pixelformat, v4l2_format_str(fmt.fmt.pix.pixelformat));
	printf("v4l2 --   color  0x%X\n", fmt.fmt.pix.colorspace);
	printf("v4l2 --   field  0x%X\n", fmt.fmt.pix.field);
}


inline void v4l2_print_formatdesc( const v4l2_fmtdesc& desc )
{
	printf("v4l2 -- format #%u\n", desc.index);
	printf("v4l2 --   desc   %s\n", desc.description);
	printf("v4l2 --   flags  %s\n", (desc.flags == 0 ? "V4L2_FMT_FLAG_UNCOMPRESSED" : "V4L2_FMT_FLAG_COMPRESSED"));
	printf("v4l2 --   fourcc 0x%X  %s\n", desc.pixelformat, v4l2_format_str(desc.pixelformat));
	
}
	

bool v4l2Camera::initFormats()
{
	struct v4l2_fmtdesc desc;
	memset(&desc, 0, sizeof(v4l2_fmtdesc));

	desc.index = 0;
	desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	while( ioctl(mFD, VIDIOC_ENUM_FMT, &desc) == 0 )
	{
		mFormats.push_back(desc);
		v4l2_print_formatdesc( desc );
		desc.index++;
	}

	return true;
}


// initStream
bool v4l2Camera::initStream()
{
	struct v4l2_format fmt;	
	memset(&fmt, 0, sizeof(v4l2_format));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	// retrieve existing video format
	if( xioctl(mFD, VIDIOC_G_FMT, &fmt) < 0 )
	{
		const int err = errno;
		printf( "v4l2 -- failed to get video format of device (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	v4l2_print_format(fmt, "preexisting format");

#if 1
	// setup new format
	struct v4l2_format new_fmt;	
	memset(&new_fmt, 0, sizeof(v4l2_format));

	new_fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	new_fmt.fmt.pix.width       = fmt.fmt.pix.width;
	new_fmt.fmt.pix.height      = fmt.fmt.pix.height;
	new_fmt.fmt.pix.pixelformat = fmt.fmt.pix.pixelformat;
	new_fmt.fmt.pix.field       = fmt.fmt.pix.field;
	new_fmt.fmt.pix.colorspace  = fmt.fmt.pix.colorspace;

	if( mRequestWidth > 0 && mRequestHeight > 0 )
	{
		new_fmt.fmt.pix.width  = mRequestWidth;
		new_fmt.fmt.pix.height = mRequestHeight;
	}

	if( mRequestFormat >= 0 && mRequestFormat < mFormats.size() )
		new_fmt.fmt.pix.pixelformat = mFormats[mRequestFormat].pixelformat;

	v4l2_print_format(new_fmt, "setting new format...");

	if( xioctl(mFD, VIDIOC_S_FMT, &new_fmt) < 0 )
	{
		const int err = errno;
		printf( "v4l2 -- failed to set video format of device (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	
	// re-retrieve the current format, with detailed info like line pitch/ect.
	memset(&fmt, 0, sizeof(v4l2_format));
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	if( xioctl(mFD, VIDIOC_G_FMT, &fmt) < 0 )
	{
		const int err = errno;
		printf( "v4l2 -- failed to get video format of device (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	v4l2_print_format(fmt, "confirmed new format");
#endif

	mWidth      = fmt.fmt.pix.width;
	mHeight     = fmt.fmt.pix.height;
	mPitch      = fmt.fmt.pix.bytesperline;
	mPixelDepth = (mPitch * 8) / mWidth;

	// initMMap
	if( !initMMap() )		// initUserPtr()
		return false;

	return true;
}


// Create
v4l2Camera* v4l2Camera::Create( const char* device_path )
{
	v4l2Camera* cam = new v4l2Camera(device_path);

	if( !cam->init() )
	{
		printf("v4l2 -- failed to create instance %s\n", device_path);
		delete cam;
		return NULL;
	}
	
	return cam;
}


// Init
bool v4l2Camera::init()
{
	// locate the /dev/event* path for this device
	mFD = open(mDevicePath.c_str(), O_RDWR | O_NONBLOCK, 0 );

	if( mFD < 0 )
	{
		printf( "v4l2 -- failed to open %s\n", mDevicePath.c_str());
		return false;
	}

	// initialize
	if( !initCaps() )
		return false;

	if( !initFormats() )
		return false;

	if( !initStream() )
		return false;

	return true;
}


// Open
bool v4l2Camera::Open()
{
	printf( "v4l2Camera::Open(%s)\n", mDevicePath.c_str());

	// begin streaming
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	printf( "v4l2 -- starting streaming %s with ioctl(VIDIOC_STREAMON)...\n", mDevicePath.c_str());

	if( xioctl(mFD, VIDIOC_STREAMON, &type) < 0 )
	{
		printf( "v4l2 -- failed to start streaming (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	return true;
}


// Close
bool v4l2Camera::Close()
{
	// stop streaming
	enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	printf( "v4l2 -- stopping streaming %s with ioctl(VIDIOC_STREAMOFF)...\n", mDevicePath.c_str());

	if( xioctl(mFD, VIDIOC_STREAMOFF, &type) < 0 )
	{
		printf( "v4l2 -- failed to stop streaming (errno=%i) (%s)\n", errno, strerror(errno));
		//return false;
	}

	return true;
}



// initCaps
bool v4l2Camera::initCaps()
{
	struct v4l2_capability caps;

	if( xioctl(mFD, VIDIOC_QUERYCAP, &caps) < 0 )
	{
		printf( "v4l2 -- failed to query caps (xioctl VIDIOC_QUERYCAP) for %s\n", mDevicePath.c_str());
		return false;
	}

	#define PRINT_CAP(x) printf( "v4l2 -- %-18s %s\n", #x, (caps.capabilities & x) ? "yes" : "no")

	PRINT_CAP(V4L2_CAP_VIDEO_CAPTURE);
	PRINT_CAP(V4L2_CAP_READWRITE);
	PRINT_CAP(V4L2_CAP_ASYNCIO);
	PRINT_CAP(V4L2_CAP_STREAMING);
	
	if( !(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE) )
	{
		printf( "v4l2 -- %s is not a video capture device\n", mDevicePath.c_str());
		return false;
	}

	return true;
}


// initUserPtr
bool v4l2Camera::initUserPtr()
{
	// request buffers
	struct v4l2_requestbuffers req;
	memset(&req, 0, sizeof(v4l2_requestbuffers));

	req.count  = REQUESTED_RINGBUFFERS;
	req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_USERPTR;

	if ( xioctl(mFD, VIDIOC_REQBUFS, &req) < 0 ) 
	{
		const int err = errno;
		printf( "v4l2 -- failed to request buffers (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}

	// queue ringbuffer
#if 0
	for( size_t n=0; n < mRingbuffer.size(); n++ )
	{
		struct v4l2_buffer buf;
		memset(&buf, 0, sizeof(v4l2_buffer));
		
		buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_USERPTR;
		buf.index  = n;
		buf.length = mRingbuffer[n]->GetSize();

		buf.m.userptr = (unsigned long)mRingbuffer[n]->GetCPU();

		if( xioctl(mFD, VIDIOC_QBUF, &buf) < 0 )
		{
			printf( "v4l2 -- failed to queue buffer %zu (errno=%i) (%s)\n", n, errno, strerror(errno));
			return false;
		}
	}
#endif

	return true;
}
