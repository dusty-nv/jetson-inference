/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __OBJECT_TRACKER_KLT_H__
#define __OBJECT_TRACKER_KLT_H__

#ifdef HAS_VPI

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>
#include <vpi/algo/KLTFeatureTracker.h>

#include "objectTracker.h"


/**
 * Object tracker using GPU-accelerated KLT from VPI library
 * https://docs.nvidia.com/vpi/algo_klt_tracker.html
 * @ingroup objectTracker
 */
class objectTrackerKLT : public objectTracker
{
public:
	/**
	 * Create a new object tracker.
	 */
	static objectTrackerKLT* Create();
	
	/**
	 * Create a new object tracker by parsing the command line.
	 */
	static objectTrackerKLT* Create( int argc, char** argv );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static objectTrackerKLT* Create( const commandLine& cmdLine );
	
	/**
	 * Destroy
	 */
	~objectTrackerKLT();
	
	/**
	 * GetType
	 */
	inline virtual Type GetType() const	{ return KLT; }

	/**
	 * Process
	 */
	virtual int Process( void* image, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections );
	
protected:
	objectTrackerKLT();
	
	bool init( uint32_t width, uint32_t height, imageFormat format );
	void free();
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint64_t mFrameCount;
	
	imageFormat mFormat;
	
	VPIImage mImages[2];
	VPIArray mInputBoxes;
	VPIArray mInputPreds;
	VPIArray mOutputBoxes;
	VPIArray mOutputPreds;
	VPIStream mStream;
	VPIPayload mPayload;
	VPIKLTFeatureTrackerParams mParams;
	
	std::vector<VPIKLTTrackedBoundingBox> mBoxes;
	std::vector<VPIHomographyTransform2D> mPreds;
	
	
};

#endif
#endif