/*
 * inference-101
 */

#ifndef __GSTREAMER_PIPELINE_H__
#define __GSTREAMER_PIPELINE_H__

#include <gst/gst.h>
#include <string>


struct _GstAppSink;
class QWaitCondition;
class QMutex;


/**
 * gstreamer generic pipeline
 * @ingroup util
 */
class gstPipeline
{
public:
	// Create pipeline
	static gstPipeline* Create( std::string pipeline, uint32_t width, uint32_t height, uint32_t depth );
	
	// Destroy
	~gstPipeline();

	// Start/stop streaming
	bool Open();
	void Close();
	
	// Capture YUV (NV12)
	bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX );
	
	// Takes in captured YUV-NV12 CUDA image, converts to float4 RGBA (with pixel intensity 0-255)
	bool ConvertRGBA( void* input, void** output );
	
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

	gstPipeline();
	
	bool init();
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

};

#endif
