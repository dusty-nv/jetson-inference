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
 
#ifdef HAS_VPI

#include "objectTrackerKLT.h"
#include "cudaColorspace.h"
#include "cudaDraw.h"


#define LOG_VPI "[VPI]    "

#define VPI_CHECK(x)   vpiCheckError((x), #x, __FILE__, __LINE__)
#define VPI_SUCCESS(x) (VPI_CHECK(x) == VPI_SUCCESS)
#define VPI_FAILED(x)  (VPI_CHECK(x) != VPI_SUCCESS)
#define VPI_VERIFY(x)  if(VPI_FAILED(x)) return false;


inline VPIStatus vpiCheckError(VPIStatus retval, const char* txt, const char* file, int line)
{
	if( retval == VPI_SUCCESS )
		return VPI_SUCCESS;
	
	char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];
	vpiGetLastStatusMessage(buffer, sizeof(buffer));
	
	LogError(LOG_VPI "error occurred at %s:%i\n", file, line);
	LogError(LOG_VPI "  %s\n", txt);
	LogError(LOG_VPI "%s\n", buffer);
	
	return retval;
}


// constructor
objectTrackerKLT::objectTrackerKLT()
{
	mWidth = 0;
	mHeight = 0;
	mFormat = IMAGE_UNKNOWN;
	mStream = NULL;
	mPayload = NULL;
	mFrameCount = 0;
	mInputBoxes = NULL;
	mInputPreds = NULL;
	mOutputBoxes = NULL;
	mOutputPreds = NULL;
	
	mBoxes.reserve(128);
	mPreds.reserve(128);
	
	memset(mImages, 0, sizeof(mImages));
}


// destructor
objectTrackerKLT::~objectTrackerKLT()
{
	free();
	vpiStreamDestroy(mStream);
}


// free
void objectTrackerKLT::free()
{
	mBoxes.clear();
	mPreds.clear();
	
	for( uint32_t n=0; n < 2; n++ )
		vpiImageDestroy(mImages[n]);
	
	vpiPayloadDestroy(mPayload);
	
	vpiArrayDestroy(mInputBoxes);
	vpiArrayDestroy(mInputPreds);
	vpiArrayDestroy(mOutputBoxes);
	vpiArrayDestroy(mOutputPreds);
}


// Create
objectTrackerKLT* objectTrackerKLT::Create()
{
	objectTrackerKLT* tracker = new objectTrackerKLT();
	
	if( !tracker )
		return NULL;
	
	if( !tracker->init(1280, 720, IMAGE_RGB8) )  // use a dummy resolution for now
	{
		LogError(LOG_VPI "failed to initialize object tracker (1280x720)\n");
		delete tracker;
		return NULL;
	}
	
	return tracker;
}


// Create
objectTrackerKLT* objectTrackerKLT::Create( const commandLine& cmdLine )
{
	return Create();
}


// Create
objectTrackerKLT* objectTrackerKLT::Create( int argc, char** argv )
{
	return Create(commandLine(argc, argv));
}

int boxesSize = 0;
int predsSize = 0;

// init
bool objectTrackerKLT::init( uint32_t width, uint32_t height, imageFormat format )
{
	if( mWidth == width && mHeight == height && mFormat == format )
		return true;
	
	if( !imageFormatIsRGB(format) ) 
	{
		imageFormatErrorMsg(LOG_VPI, "objectTrackerKLT::init", format);
		return false;
	}
	
	// free previous resources if init() was called before
	free();
	
	// the stream only needs to be created once
	if( !mStream )
		VPI_VERIFY(vpiStreamCreate(VPI_BACKEND_CUDA, &mStream));

	// allocate resources
	for( uint32_t n=0; n < 2; n++ )
		VPI_VERIFY(vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8, VPI_BACKEND_CUDA|VPI_EXCLUSIVE_STREAM_ACCESS, &mImages[n]));

	VPI_VERIFY(vpiInitKLTFeatureTrackerParams(&mParams));
	
	//mParams.nccThresholdKill = 0.85f;
	//mParams.nccThresholdUpdate = 0.95f;
	
	VPIKLTFeatureTrackerCreationParams creationParams;
	
	creationParams.maxTemplateCount = 128;
	creationParams.maxTemplateWidth = 128;
	creationParams.maxTemplateHeight = 128;
	
	VPI_VERIFY(vpiCreateKLTFeatureTracker(VPI_BACKEND_CUDA, width, height, VPI_IMAGE_FORMAT_U8, &creationParams, &mPayload));
	
	VPI_VERIFY(vpiArrayCreate(128, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &mOutputBoxes));
	VPI_VERIFY(vpiArrayCreate(128, VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D, 0, &mOutputPreds));
	
	VPIArrayData wrapper = {};
	//int wrapperSize = 0;

#if 1 // VPI_VERSION >= 2
	wrapper.bufferType             = VPI_ARRAY_BUFFER_HOST_AOS;
	wrapper.buffer.aos.type        = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
	wrapper.buffer.aos.capacity    = mBoxes.capacity();
	wrapper.buffer.aos.sizePointer = &boxesSize;
	wrapper.buffer.aos.data        = &mBoxes[0];
	
	VPI_VERIFY(vpiArrayCreateWrapper(&wrapper, 0, &mInputBoxes));
	
	wrapper.buffer.aos.type        = VPI_ARRAY_TYPE_HOMOGRAPHY_TRANSFORM_2D;
	wrapper.buffer.aos.sizePointer = &predsSize;
	wrapper.buffer.aos.data        = &mPreds[0];
		  
	VPI_VERIFY(vpiArrayCreateWrapper(&wrapper, 0, &mInputPreds));
	
#else
	wrapper.type         = VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX;
	wrapper.capacity     = bboxes.capacity();
	wrapper.sizePointer  = &bboxesSize;
	wrapper.data         = &bboxes[0];
#endif		  
	
	mWidth = width;
	mHeight = height;
	mFormat = format;
	mFrameCount = 0;
	
	return true;
}

	
// macro for converting VPI bounding box to (x1, y1, x2, y2)
#define UNPACK_BOX(box) \
	const float x1 = box.bbox.xform.mat3[0][2];	\
	const float y1 = box.bbox.xform.mat3[1][2];	\
	const float x2 = box.bbox.xform.mat3[0][0] * box.bbox.width + x1;  \
	const float y2 = box.bbox.xform.mat3[1][1] * box.bbox.height + y1;
	
#define UNPACK_BOX_PRED(box, pred) \
	const float x1 = box.bbox.xform.mat3[0][2] + pred.mat3[0][2];	\
	const float y1 = box.bbox.xform.mat3[1][2] + pred.mat3[1][2];	\
	const float x2 = box.bbox.xform.mat3[0][0] * pred.mat3[0][0] * box.bbox.width + x1;  \
	const float y2 = box.bbox.xform.mat3[1][1] * pred.mat3[1][1] * box.bbox.height + y1;
	
	
// debug info about VPI bounding boxes
void printBox( const VPIKLTTrackedBoundingBox& box )
{
	LogVerbose(LOG_VPI "trackingStatus=%i  templateStatus=%i\n", box.trackingStatus, box.templateStatus);
	LogVerbose(LOG_VPI "width=%f  height=%f\n", box.bbox.width, box.bbox.height);
	
	for( int n=0; n < 3; n++ )
		LogVerbose(LOG_VPI "[ %f %f %f ]\n", box.bbox.xform.mat3[n][0], box.bbox.xform.mat3[n][1], box.bbox.xform.mat3[n][2]);
}
	
	
// debug info about VPI array	
void printArray( const VPIArrayData& info, const char* prefix="" )
{
	LogVerbose(LOG_VPI "%s  size=%i  capacity=%i  stride=%i  type=%i  bufferType=%i\n", prefix, *info.buffer.aos.sizePointer, info.buffer.aos.capacity, info.buffer.aos.strideBytes, (int)info.buffer.aos.type, (int)info.bufferType);
}
	
	
// find the VPI bounding box with the maximum overlap with the detection
int findBox( const detectNet::Detection& detection, VPIKLTTrackedBoundingBox* boxes, VPIHomographyTransform2D* preds, int numBoxes, float overlapThreshold=0.0f )
{
	if( !boxes || numBoxes <= 0 )
		return -1;
	
	int maxBox = -1;
	float maxOverlap = 0.0f;
	
	const float detectionArea = detection.Area();
	
	for( int n=0; n < numBoxes; n++ )
	{
		if( boxes[n].trackingStatus )
			continue;
		
		UNPACK_BOX_PRED(boxes[n], preds[n]);
		
		const float area = detectNet::Detection::Area(x1, y1, x2, y2);
		const float overlap = detection.IntersectionArea(x1, y1, x2, y2) / fmaxf(detectionArea, area);
		
		if( overlap > overlapThreshold && overlap > maxOverlap )
		{
			maxBox = n;
			maxOverlap = overlap;
		}
	}
	
	return maxBox;
}

		
// find the detection with the maximum overlap with the VPI bounding box
int findDetection( const VPIKLTTrackedBoundingBox& box, const VPIHomographyTransform2D& pred, detectNet::Detection* detections, int numDetections, float overlapThreshold=0.0f )
{
	if( !detections || numDetections <= 0 )
		return -1;
	
	UNPACK_BOX_PRED(box, pred);
	
	const float area = detectNet::Detection::Area(x1, y1, x2, y2);
	
	int maxDetection = -1;
	float maxOverlap = 0.0f;
	
	for( int n=0; n < numDetections; n++ )
	{
		const float overlap = detections[n].IntersectionArea(x1, y1, x2, y2) / fmaxf(detections[n].Area(), area);
		
		if( overlap > overlapThreshold && overlap > maxOverlap )
		{
			maxDetection = n;
			maxOverlap = overlap;
		}
	}
	
	return maxDetection;
}


// convert a detectNet bounding box to a VPI bounding box
void detectionToBox( const detectNet::Detection& detection, VPIKLTTrackedBoundingBox& box )
{
	memset(box.bbox.xform.mat3, 0, sizeof(box.bbox.xform.mat3));
	
	box.bbox.xform.mat3[0][0] = 1;
	box.bbox.xform.mat3[1][1] = 1;
	box.bbox.xform.mat3[2][2] = 1;
	
	box.bbox.xform.mat3[0][2] = detection.Left;
	box.bbox.xform.mat3[1][2] = detection.Top;
	
	box.bbox.width = detection.Width();
	box.bbox.height = detection.Height();
}


// Process
int objectTrackerKLT::Process( void* input, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections )
{
	if( !mEnabled )
		return numDetections;
	
	if( !init(width, height, format) )
	{
		LogError(LOG_VPI "failed to initialize object tracker (%ux%u)\n", width, height);
		return -1;
	}
	
	VPIImage currImage = mImages[mFrameCount % 2];
	VPIImage lastImage = mImages[(mFrameCount + 1) % 2];
	
	// copy the latest frame into the VPI circular buffer
	// the incoming images have an unknown lifespan so best not to cache them directly
	VPIImageData imgData = {};
	
	VPI_VERIFY(vpiImageLockData(currImage, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR, &imgData));
	CUDA_VERIFY(cudaConvertColor(input, format, imgData.buffer.pitch.planes[0].data, IMAGE_GRAY8, width, height));
	VPI_VERIFY(vpiImageUnlock(currImage));
	
	if( mFrameCount == 0 )
	{
		mFrameCount++;
		return 0;
	}
	
	// update the tracker
	VPI_VERIFY(vpiSubmitKLTFeatureTracker(mStream, VPI_BACKEND_CUDA, mPayload, lastImage, mInputBoxes, mInputPreds, currImage, mOutputBoxes, mOutputPreds, &mParams));
	VPI_VERIFY(vpiStreamSync(mStream));
	
	// lock data 
	VPI_VERIFY(vpiArrayLock(mInputBoxes, VPI_LOCK_READ_WRITE));
	VPI_VERIFY(vpiArrayLock(mInputPreds, VPI_LOCK_READ_WRITE));
	
	VPIArrayData outputBoxData;
	VPIArrayData outputPredData;
	
     VPI_VERIFY(vpiArrayLockData(mOutputBoxes, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outputBoxData));
	VPI_VERIFY(vpiArrayLockData(mOutputPreds, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outputPredData));

	VPIKLTTrackedBoundingBox* updatedBoxes = (VPIKLTTrackedBoundingBox*)outputBoxData.buffer.aos.data;
	VPIHomographyTransform2D* updatedPreds = (VPIHomographyTransform2D*)outputPredData.buffer.aos.data;

	printArray(outputBoxData, "outputBoxData");
	printArray(outputPredData, "outputPredData");
	
	// update data
	const int numBoxes = mBoxes.size();
	
	for( int n=0; n < numBoxes; n++ )
	{
		// if tracking failed, update the input bbox's tracking status too
		if( updatedBoxes[n].trackingStatus )
		{
			if( mBoxes[n].trackingStatus == 0 )
			{
				mBoxes[n].trackingStatus = 1;
				LogVerbose(LOG_VPI "dropped track %i\n", n);
			}
			
			continue;
		}
		
		// does the input bbox need updating?
		if( updatedBoxes[n].templateStatus )
		{
			// search detections list for update
			const int d = findDetection(updatedBoxes[n], updatedPreds[n], detections, numDetections);
			
			if( d >= 0 )
			{
				detectionToBox(detections[d], mBoxes[n]);
				LogVerbose(LOG_VPI "updated track %i with detection (%f, %f, %f, %f) classID=%u\n", n, detections[d].Left, detections[d].Top, detections[d].Right, detections[d].Bottom, detections[d].ClassID);
			}
			else
			{
				mBoxes[n] = updatedBoxes[n];
				LogVerbose(LOG_VPI "updated track %i with output bbox from tracker\n", n);
			}
			
			mBoxes[n].templateStatus = 1;
			
			mPreds[n] = VPIHomographyTransform2D{};
			mPreds[n].mat3[0][0] = 1;
			mPreds[n].mat3[1][1] = 1;
			mPreds[n].mat3[2][2] = 1;
		}
		else
		{
			mBoxes[n].templateStatus = 0;
			mPreds[n] = updatedPreds[n];
		}
		
	#if 1
		UNPACK_BOX_PRED(updatedBoxes[n], updatedPreds[n]);
		LogVerbose(LOG_VPI "track %i  trackingStatus=%i  templateStatus=%i  (%f, %f, %f, %f)\n", n, updatedBoxes[n].trackingStatus, updatedBoxes[n].templateStatus, x1, y1, x2, y2);
		//printBox(updatedBoxes[n]);
	#endif
	}
	
	// add new detections
	for( int n=0; n < numDetections; n++ )
	{
		const int b = findBox(detections[n], updatedBoxes, updatedPreds, numBoxes);
		
		if( b < 0 && mBoxes.size() < mBoxes.capacity() )
		{
			VPIKLTTrackedBoundingBox box = {};
			
			detectionToBox(detections[n], box);
			
			box.trackingStatus = 0;
			box.templateStatus = 1;
			
			VPIHomographyTransform2D xform = {};
			xform.mat3[0][0] = 1;
			xform.mat3[1][1] = 1;
			xform.mat3[2][2] = 1;
			 
			mBoxes.push_back(box);
			mPreds.push_back(xform);
			
			VPI_VERIFY(vpiArraySetSize(mInputBoxes, mBoxes.size()));
			VPI_VERIFY(vpiArraySetSize(mInputPreds, mPreds.size()));
			
			LogVerbose(LOG_VPI "added track %zu from detection (%f, %f, %f, %f) classID=%u\n", mBoxes.size()-1, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].ClassID);
		}
	}
	
	// draw tracked boxes
	for( int n=0; n < numBoxes; n++ )
	{
		if( updatedBoxes[n].trackingStatus )
			continue;
		
		UNPACK_BOX_PRED(updatedBoxes[n], updatedPreds[n]);
		CUDA(cudaDrawRect(input, width, height, format, x1, y1, x2, y2, make_float4(0,0,0,0), make_float4(255,255,255,255), 1.0f));
	}
	
	// unlock data
	VPI_VERIFY(vpiArrayUnlock(mInputBoxes));
	VPI_VERIFY(vpiArrayUnlock(mInputPreds));
	VPI_VERIFY(vpiArrayUnlock(mOutputBoxes));
	VPI_VERIFY(vpiArrayUnlock(mOutputPreds));
	
	mFrameCount++;
	return 0;	// TODO update output tracks
}

#endif