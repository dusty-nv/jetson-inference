/*
 * Test your camera if you have issues:
 *     gst-launch-1.0 v4l2src device=/dev/video1 ! video/x-raw, width=1280, height=720 ! xvimagesink
 */

#ifndef __CONFIG_H__
#define __CONFIG_H__


#define VIDEO_NV                      0  // Original NV CSI camera
#define VIDEO_GST_RTP_SRC             1  // Gstreamer udpsrc
#define VIDEO_GST_V4L_SRC             2  // Gstreamer v4l2src
#define VIDEO_RTP_STREAM_SOURCE       3  // Raw RTP
#define VIDEO_GV_STREAM_SOURCE        4  // GigE Vision (Aravis)

/*
 * This is the important bit select your video source
 */
#define VIDEO_SRC                   VIDEO_GST_V4L_SRC
//#define VIDEO_SRC                     VIDEO_GV_STREAM_SOURCE

#if VIDEO_SRC==VIDEO_GST_RTP_SRC
#define VIDEO_SRC_NAME                "Gstreamer RTP"
#endif
#if VIDEO_SRC==VIDEO_RTP_STREAM_SOURCE
#define VIDEO_SRC_NAME                "Real Time Protocol (RTP)"
#endif
#if VIDEO_SRC==VIDEO_GST_V4L_SRC
#define VIDEO_SRC_NAME                "Gstreamer V4L2"
#endif
#if VIDEO_SRC==VIDEO_GV_STREAM_SOURCE
#define VIDEO_SRC_NAME                "GigEVision"
#endif
#if VIDEO_SRC==VIDEO_NV
#define VIDEO_SRC_NAME                "Nvidia CSI"
#endif

#define SDL_DISPLAY                   1  // Use SDL for video display
#define GST_RTP_SINK                  0  // Enable the RTP output of rendered stream
#define LOGO                          1  // Display some branding
#define COMPANY_NAME                  "Ross Newman"
#define COMPANY_URL                   "rossnewman.com"
#define COMPANY_LOGO                  "logo.bmp"

#if 1
#define WIDTH                         1280 // 720p default for HD Webcam
#define HEIGHT                        720
#else
#define WIDTH                         1920 // Max resolution for ptGrey GigEVision Blackfly is 1920 x 1200
#define HEIGHT                        1080 
#endif

//
// GigEVision camera settings
//
#define VIDEO_GV_NULL                 0 // Not used
#define VIDEO_GV_BAYER_GR8            1 // ARV_PIXEL_FORMAT_BAYER_GR_8
#define VIDEO_GV_YUV422               2 // ARV_PIXEL_FORMAT_YUV_422_PACKED
#define VIDEO_GV_RGB8                 3 // ARV_PIXEL_FORMAT_RGB_8_PACKED
#define VIDEO_GV_PIXEL_FORMAT         VIDEO_GV_YUV422
#define VIDEO_GV_SRC_FRAMERATE        30.0

//
// Gstreamer V4L2 settings
//
#define VIDEO_GST_V4L_SRC_DEVICE      "/dev/video0" // Note '/dev/video0' is the CSI camera on the TX dev platforms
#define VIDEO_GST_V4L_SRC_FRAMERATE   30

#if VIDEO_SRC == VIDEO_GV_STREAM_SOURCE
#define VIDEO_DEFAULT_FRAMERATE       VIDEO_GV_SRC_FRAMERATE // GigE Vision (Aravis)
#else
#define VIDEO_DEFAULT_FRAMERATE       VIDEO_GST_V4L_SRC_FRAMERATE 
#endif


#if VIDEO_GV_PIXEL_FORMAT==VIDEO_GV_BAYER_GR8
#define VIDEO_BYTES_PER_PIXEL         1
#define VIDEO_GV_PIXEL_FORMAT_NAME    " BAYER GR8"
#elif VIDEO_GV_PIXEL_FORMAT==VIDEO_GV_YUV422
#define VIDEO_BYTES_PER_PIXEL         2
#define VIDEO_GV_PIXEL_FORMAT_NAME    " YUV422"
#elif VIDEO_GV_PIXEL_FORMAT==VIDEO_GV_RGB8
#define VIDEO_BYTES_PER_PIXEL         3  
#define VIDEO_GV_PIXEL_FORMAT_NAME    " RGB8"
#else
#define VIDEO_BYTES_PER_PIXEL         2  
#define VIDEO_GV_PIXEL_FORMAT_NAME    ""
#endif


//
// detectnet options
//
#define DETECTNET_TYPE_PED            detectNet::PEDNET // Pedestrians and Luggage
#define DETECTNET_TYPE_FACE           detectNet::FACENET 
#define DETECTNET_TYPE                DETECTNET_TYPE_PED // network type

//
// RTP Connection details
//
#define IP_UNICAST                    "127.0.0.1"
#define IP_MULTICAST_OUT              "239.192.1.198"
#define IP_PORT_OUT                   5004
#define IP_MULTICAST_IN               "239.192.2.34"
#define IP_PORT_IN                    5004

//
// Gstreamer1.0 compatability
//
#define GST_1_FUDGE                   1       // Offset date by 1 byte as gstream has an RTP bug. Must have RTP_STREAM_SOURCE=1
#endif
