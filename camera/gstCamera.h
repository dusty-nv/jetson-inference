/*
 * inference-101
 */

#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <gst/gst.h>
#include <string>
#include "camera.h"


struct _GstAppSink;
class QWaitCondition;
class QMutex;


/**
 * gstreamer CSI camera using nvcamerasrc
 */
class gstCamera : public camera
{
public:
	static gstCamera* Create();
	static gstCamera* Create(std::string pipeline, int height, int width);

	gstCamera(int height, int width);
	~gstCamera();

	bool Open();

	void Close();
	
	// Capture YUV (NV12)
	bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX );
	
private:
	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	gstCamera();
	
	bool init(std::string pipeline);
	bool buildLaunchStr(std::string pipeline);
	void checkMsgBus();
	void checkBuffer();
	
	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;

	std::string  mLaunchStr;
	
 	static bool mOnboardCamera;
	static const uint32_t NUM_RINGBUFFERS = 4;
	
	void* mRingbufferCPU[NUM_RINGBUFFERS];
	void* mRingbufferGPU[NUM_RINGBUFFERS];
	
	QWaitCondition* mWaitEvent;
	
	QMutex* mWaitMutex;
	QMutex* mRingMutex;
	
	uint32_t mLatestRingbuffer;
	bool     mLatestRetrieved;
};

#endif
