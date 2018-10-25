/*
 * inference-101
 */

#include "gstPipeline.h"
#include "gstUtility.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <sstream> 
#include <unistd.h>
#include <string.h>

#include <QMutex>
#include <QWaitCondition>

#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"



// constructor
gstPipeline::gstPipeline()
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;	

	mWidth  = 0;
	mHeight = 0;
	mDepth  = 0;
	mSize   = 0;
	
	mWaitEvent  = new QWaitCondition();
	mWaitMutex  = new QMutex();
	mRingMutex  = new QMutex();
	
	mLatestRGBA       = 0;
	mLatestRingbuffer = 0;
	mLatestRetrieved  = false;
	
	for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
	{
		mRingbufferCPU[n] = NULL;
		mRingbufferGPU[n] = NULL;
		mRGBA[n]          = NULL;
	}
}


// destructor	
gstPipeline::~gstPipeline()
{
	
}


// ConvertRGBA
bool gstPipeline::ConvertRGBA( void* input, void** output )
{
	if( !input || !output )
		return false;
	
	if( !mRGBA[0] )
	{
		for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
		{
			if( CUDA_FAILED(cudaMalloc(&mRGBA[n], mWidth * mHeight * sizeof(float4))) )
			{
				printf(LOG_CUDA "gstPipeline -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
				return false;
			}
		}
		
		printf(LOG_CUDA "gstreamer pipeline -- allocated %u RGBA ringbuffers\n", NUM_RINGBUFFERS);
	}
	
	if( mDepth == 12 )
	{
		// NV12
		if( CUDA_FAILED(cudaNV12ToRGBAf((uint8_t*)input, (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)) )
			return false;
	}
	else
	{
		// RGB
		if( CUDA_FAILED(cudaRGBToRGBAf((uchar3*)input, (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)) )
			return false;
	}
	
	*output     = mRGBA[mLatestRGBA];
	mLatestRGBA = (mLatestRGBA + 1) % NUM_RINGBUFFERS;
	return true;
}


// onEOS
void gstPipeline::onEOS(_GstAppSink* sink, void* user_data)
{
	printf(LOG_GSTREAMER "gstreamer decoder onEOS\n");
}


// onPreroll
GstFlowReturn gstPipeline::onPreroll(_GstAppSink* sink, void* user_data)
{
	printf(LOG_GSTREAMER "gstreamer decoder onPreroll\n");
	return GST_FLOW_OK;
}


// onBuffer
GstFlowReturn gstPipeline::onBuffer(_GstAppSink* sink, void* user_data)
{
	//printf(LOG_GSTREAMER "gstreamer decoder onBuffer\n");
	
	if( !user_data )
		return GST_FLOW_OK;

	gstPipeline* dec = (gstPipeline*)user_data;
	
	dec->checkBuffer();
	dec->checkMsgBus();
	return GST_FLOW_OK;
}
	

// Capture
bool gstPipeline::Capture( void** cpu, void** cuda, unsigned long timeout )
{
	mWaitMutex->lock();
    const bool wait_result = mWaitEvent->wait(mWaitMutex, timeout);
    mWaitMutex->unlock();
	
	if( !wait_result )
		return false;
	
	mRingMutex->lock();
	const uint32_t latest = mLatestRingbuffer;
	const bool retrieved = mLatestRetrieved;
	mLatestRetrieved = true;
	mRingMutex->unlock();
	
	// skip if it was already retrieved
	if( retrieved )
		return false;
	
	if( cpu != NULL )
		*cpu = mRingbufferCPU[latest];
	
	if( cuda != NULL )
		*cuda = mRingbufferGPU[latest];
	
	return true;
}


#define release_return { gst_sample_unref(gstSample); return; }


// checkBuffer
void gstPipeline::checkBuffer()
{
	if( !mAppSink )
		return;

	// block waiting for the buffer
	GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
	
	if( !gstSample )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_sample_get_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve
	GstMapInfo map; 

	if(	!gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_buffer_map() failed...\n");
		return;
	}
	
	//gst_util_dump_mem(map.data, map.size); 

	void* gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
	const uint32_t gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_buffer had NULL caps...\n");
		release_return;
	}
	
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_caps had NULL structure...\n");
		release_return;
	}
	
	// get width & height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		printf(LOG_GSTREAMER "gstreamer pipeline -- gst_caps missing width/height...\n");
		release_return;
	}
	
	if( width < 1 || height < 1 )
		release_return;
	
	mWidth  = width;
	mHeight = height;
	mDepth  = (gstSize * 8) / (width * height);
	mSize   = gstSize;
	
	//printf(LOG_GSTREAMER "gstreamer pipeline recieved %ix%i frame (%u bytes, %u bpp)\n", width, height, gstSize, mDepth);
	
	// make sure ringbuffer is allocated
	if( !mRingbufferCPU[0] )
	{
		for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
		{
			if( !cudaAllocMapped(&mRingbufferCPU[n], &mRingbufferGPU[n], gstSize) )
				printf(LOG_CUDA "gstreamer pipeline -- failed to allocate ringbuffer %u  (size=%u)\n", n, gstSize);
		}
		
		printf(LOG_CUDA "gstreamer pipeline -- allocated %u ringbuffers, %u bytes each\n", NUM_RINGBUFFERS, gstSize);
	}
	
	// copy to next ringbuffer
	const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;		
	
	//printf(LOG_GSTREAMER "gstreamer pipeline -- using ringbuffer #%u for next frame\n", nextRingbuffer);
	memcpy(mRingbufferCPU[nextRingbuffer], gstData, gstSize);
	gst_buffer_unmap(gstBuffer, &map); 
	//gst_buffer_unref(gstBuffer);
	gst_sample_unref(gstSample);
	
	
	// update and signal sleeping threads
	mRingMutex->lock();
	mLatestRingbuffer = nextRingbuffer;
	mLatestRetrieved  = false;
	mRingMutex->unlock();
	mWaitEvent->wakeAll();
}

// Create
gstPipeline* gstPipeline::Create( std::string pipeline, uint32_t width, uint32_t height, uint32_t depth )
{
	if( !gstreamerInit() )
	{
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	gstPipeline* pipe = new gstPipeline();
	
	if( !pipe )
		return NULL;

	pipe->mWidth      = width;
	pipe->mHeight     = height;
	pipe->mDepth      = depth;	// NV12 or RGB
	pipe->mSize       = (width * height * pipe->mDepth) / 8;
	pipe->mLaunchStr  = pipeline;

	if( !pipe->init() )
	{
		printf(LOG_GSTREAMER "failed to init gstPipeline\n");
		return NULL;
	}
	
	return pipe;
}

// init
bool gstPipeline::init()
{
	GError* err = NULL;

	// launch pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		printf(LOG_GSTREAMER "gstreamer decoder failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		printf(LOG_GSTREAMER "gstreamer failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		printf(LOG_GSTREAMER "gstreamer failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		printf(LOG_GSTREAMER "gstreamer failed to retrieve AppSink element from pipeline\n");
		return false;
	}
	
	mAppSink = appsink;
	
	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;
	cb.new_sample  = onBuffer;
	
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	return true;
}


// Open
bool gstPipeline::Open()
{
	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "gstreamer transitioning pipeline to GST_STATE_PLAYING\n");
	
	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_PLAYING);

	if( result == GST_STATE_CHANGE_ASYNC )
	{
#if 0
		GstMessage* asyncMsg = gst_bus_timed_pop_filtered(mBus, 5 * GST_SECOND, 
    	 					      (GstMessageType)(GST_MESSAGE_ASYNC_DONE|GST_MESSAGE_ERROR)); 

		if( asyncMsg != NULL )
		{
			gst_message_print(mBus, asyncMsg, this);
			gst_message_unref(asyncMsg);
		}
		else
			printf(LOG_GSTREAMER "gstreamer NULL message after transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS )
	{
		printf(LOG_GSTREAMER "gstreamer failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100*1000);
	checkMsgBus();

	return true;
}
	

// Close
void gstPipeline::Close()
{
	// stop pipeline
	printf(LOG_GSTREAMER "gstreamer transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstreamer failed to set pipeline state to PLAYING (error %u)\n", result);

	usleep(250*1000);
}


// checkMsgBus
void gstPipeline::checkMsgBus()
{
	while(true)
	{
		GstMessage* msg = gst_bus_pop(mBus);

		if( !msg )
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
}
