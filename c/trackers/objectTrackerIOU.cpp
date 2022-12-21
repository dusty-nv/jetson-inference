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

#include "objectTrackerIOU.h"


// constructor
objectTrackerIOU::objectTrackerIOU()
{
	mFrameCount = 0;
	mInstanceCount = 0;
	
	mTracks.reserve(128);
}


// destructor
objectTrackerIOU::~objectTrackerIOU()
{

}

// Create
objectTrackerIOU* objectTrackerIOU::Create()
{
	objectTrackerIOU* tracker = new objectTrackerIOU();
	
	if( !tracker )
		return NULL;

	return tracker;
}


// Create
objectTrackerIOU* objectTrackerIOU::Create( const commandLine& cmdLine )
{
	return Create();
}


// Create
objectTrackerIOU* objectTrackerIOU::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}


// find the detection with the highest IOU that exceeds the given threshold
int findBestIOU( const detectNet::Detection& track, detectNet::Detection* detections, int numDetections, float threshold=0.5f )
{
	int maxDetection = -1;
	float maxIOU = 0.0f;
	
	for( int n=0; n < numDetections; n++ )
	{
		if( detections[n].Instance >= 0 )
			continue; // this bbox is already a match for another track
		
		if( detections[n].ClassID != track.ClassID )
			continue;
		
		const float IOU = track.IOU(detections[n]);
		
		if( IOU >= threshold && IOU > maxIOU )
		{
			maxIOU = IOU;
			maxDetection = n;
		}
	}
	
	return maxDetection;
}
		
		
// Process
int objectTrackerIOU::Process( void* input, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections )
{
	// update active tracks
	for( int n=0; n < mTracks.size(); n++ )
	{
		const int bestMatch = findBestIOU(mTracks[n], detections, numDetections);
		
		if( bestMatch >= 0 )
		{
			detections[bestMatch].Instance = mTracks[n].Instance;
			detections[bestMatch].TrackFrames = mTracks[n].TrackFrames + 1;
			detections[bestMatch].TrackLost = 0;
			
			mTracks[n] = detections[bestMatch];
			
			LogVerbose(LOG_TRACKER "updated track -> instance=%i class=%u frames=%i\n", detections[n].Instance, detections[n].ClassID, detections[n].TrackFrames);
		}
		else
		{
			mTracks[n].TrackLost++;
		}
	}
	
	// add new tracks
	for( int n=0; n < numDetections; n++ )
	{
		if( detections[n].Instance >= 0 )
			continue;
		
		detections[n].Instance = mInstanceCount++;
		detections[n].TrackFrames = 0;
		detections[n].TrackLost = 0;
		
		mTracks.push_back(detections[n]);
		
		LogVerbose(LOG_TRACKER "added track -> instance=%i class=%u\n", detections[n].Instance, detections[n].ClassID);
	}
	
	// add valid tracks to the output array
	numDetections = 0;
	
	for( int n=0; n < mTracks.size(); n++ )
	{
		if( mTracks[n].TrackFrames >= 3 )
			detections[numDetections++] = mTracks[n];
	}
	
	// remove dropped tracks
	for( auto iter = mTracks.begin(); iter != mTracks.end(); )
	{
		if( iter->TrackLost > 15 )
		{
			LogVerbose(LOG_TRACKER "dropped track -> instance=%i class=%u frames=%i\n", iter->Instance, iter->ClassID, iter->TrackFrames);
			iter = mTracks.erase(iter);
		}
		else
		{
			++iter;
		}
	}
	
	mFrameCount++;
	return numDetections;
}
