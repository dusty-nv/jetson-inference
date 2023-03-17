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
 
#ifndef __OBJECT_TRACKER_IOU_H__
#define __OBJECT_TRACKER_IOU_H__


#include "objectTracker.h"


/**
 * The number of re-identified frames before establishing a track
 * @ingroup objectTracker
 */
#define OBJECT_TRACKER_DEFAULT_MIN_FRAMES 3

/**
 * The number of consecutive lost frames after which a track is removed
 * @ingroup objectTracker
 */
#define OBJECT_TRACKER_DEFAULT_DROP_FRAMES 15

/**
 * How much IOU overlap is required for a bounding box to be matched
 */
#define OBJECT_TRACKER_DEFAULT_OVERLAP_THRESHOLD 0.5


/**
 * Object tracker using Intersection-Over-Union (IOU)
 *
 * "High-Speed Tracking-by-Detection Without Using Image Information"
 * http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf
 *
 * This tracker essentially performs temporal clustering of bounding boxes
 * without using visual information, hence it is very fast but low accuracy.
 *
 * @ingroup objectTracker
 */
class objectTrackerIOU : public objectTracker
{
public:
	/**
	 * Create a new object tracker.
	 * @param minFrames the number of re-identified frames before before establishing a track
	 * @param dropFrames the number of consecutive lost frames after which a track is removed
	 */
	static objectTrackerIOU* Create( uint32_t minFrames=OBJECT_TRACKER_DEFAULT_MIN_FRAMES,
							   uint32_t dropFrames=OBJECT_TRACKER_DEFAULT_DROP_FRAMES,
							   float overlapThreshold=OBJECT_TRACKER_DEFAULT_OVERLAP_THRESHOLD );
	
	/**
	 * Create a new object tracker by parsing the command line.
	 */
	static objectTrackerIOU* Create( int argc, char** argv );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static objectTrackerIOU* Create( const commandLine& cmdLine );
	
	/**
	 * Destroy
	 */
	virtual ~objectTrackerIOU();
	
	/**
	 * The number of re-identified frames before before establishing a track
	 */
	inline uint32_t GetMinFrames() const					{ return mMinFrames; }
	
	/**
	 * Set the number of re-identified frames before before establishing a track
	 */
	inline void SetMinFrames( uint32_t frames )				{ mMinFrames = frames; }
	
	/**
	 * The number of consecutive lost frames after which a track is removed
	 */
	inline uint32_t GetDropFrames() const					{ return mDropFrames; }
	
	/**
	 * Set the number of consecutive lost frames after which a track is removed
	 */
	inline void SetDropFrames( uint32_t frames )				{ mDropFrames = frames; }
	
	/**
	 * How much IOU overlap is required for a bounding box to be matched
	 */
	inline float GetOverlapThreshold() const				{ return mOverlapThreshold; }
	
	/**
	 * Set how much IOU overlap is required for a bounding box to be matched
	 */
	inline void SetOverlapThreshold( float threshold )		{ mOverlapThreshold = threshold; }
	
	/**
	 * @see objectTracker::GetType
	 */
	inline virtual Type GetType() const					{ return IOU; }
	
	/**
	 * @see objectTracker::Process
	 */
	virtual int Process( void* image, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections );
	
protected:
	objectTrackerIOU( uint32_t minFrames, uint32_t dropFrames, float overlapThreshold );
	
	uint32_t mIDCount;
	uint64_t mFrameCount;
	
	uint32_t mMinFrames;
	uint32_t mDropFrames;
	
	float mOverlapThreshold;

	std::vector<detectNet::Detection> mTracks;
};

#endif