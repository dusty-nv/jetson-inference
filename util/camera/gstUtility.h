/*
 * inference-101
 */

#ifndef __GSTREAMER_UTILITY_H__
#define __GSTREAMER_UTILITY_H__


#include <gst/gst.h>


/**
 * LOG_GSTREAMER printf prefix
 */
#define LOG_GSTREAMER "[gstreamer] "


bool gstreamerInit();

gboolean gst_message_print(_GstBus* bus, _GstMessage* message, void* user_data);



#endif

