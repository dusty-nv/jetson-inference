/*
 * inference-101
 */

#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <gst/gst.h>
#include <string>


struct _GstAppSink;
class QWaitCondition;
class QMutex;


/**
 * gstreamer CSI camera using nvcamerasrc (or optionally v4l2src)
 * @ingroup util
 */
class gstCamera
{
public:
	// Create camera
	static gstCamera* Create( int v4l2_device=-1 );	// use onboard camera by default (>=0 for V4L2)
	static gstCamera* Create( uint32_t width, uint32_t height, int v4l2_device=-1 );
	
	// Destroy
	~gstCamera();

	// Start/stop streaming
	bool Open();
	void Close();
	
	// Capture YUV (NV12)
	bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX );
	
	// Takes in captured YUV-NV12 CUDA image, converts to float4 RGBA (with pixel intensity 0-255)
	// Set zeroCopy to true if you need to access ConvertRGBA from CPU, otherwise it will be CUDA only.
	bool ConvertRGBA( void* input, void** output, bool zeroCopy=false );
	
	// Image dimensions
	inline uint32_t GetWidth() const	  { return mWidth; }
	inline uint32_t GetHeight() const	  { return mHeight; }
	inline uint32_t GetPixelDepth() const { return mDepth; }
	inline uint32_t GetSize() const		  { return mSize; }
	
	// Default resolution, unless otherwise specified during Create()
	static const uint32_t DefaultWidth  = 1280;
	static const uint32_t DefaultHeight = 720;
	
private:
	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	gstCamera();
	
	bool init();
	bool buildLaunchStr();
	void checkMsgBus();
	void checkBuffer();
	
	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;

	std::string  mLaunchStr;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mDepth;
	uint32_t mSize;
	
	static const uint32_t NUM_RINGBUFFERS = 16;
	
	void* mRingbufferCPU[NUM_RINGBUFFERS];
	void* mRingbufferGPU[NUM_RINGBUFFERS];
	
	QWaitCondition* mWaitEvent;
	
	QMutex* mWaitMutex;
	QMutex* mRingMutex;
	
	uint32_t mLatestRGBA;
	uint32_t mLatestRingbuffer;
	bool     mLatestRetrieved;
	
	void* mRGBA[NUM_RINGBUFFERS];
	int   mV4L2Device;	// -1 for onboard, >=0 for V4L2 device
	
	inline bool onboardCamera() const		{ return (mV4L2Device < 0); }
};

#endif
