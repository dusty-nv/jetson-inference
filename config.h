#define VIDEO_NV 		 		0  // Original NV CSI camera
#define VIDEO_GST_RTP_SRC 		1  // Gstreamer udpsrc
#define VIDEO_GST_V4L_SRC		2  // Gstreamer v4l2src
#define VIDEO_RTP_STREAM_SOURCE	3  // Raw RTP

#define VIDEO_SRC				VIDEO_GST_V4L_SRC
#define SDL_DISPLAY 			1
#define GST_RTP_SINK			0
#define ABACO 					1

#define HEIGHT 					720
#define WIDTH 					1280

// 
// Connection details
//
#define IP_UNICAST          "127.0.0.1"
#define IP_MULTICAST_OUT    "239.192.1.198"
#define IP_PORT_OUT    		5004
#define IP_MULTICAST_IN     "239.192.2.34"
#define IP_PORT_IN     		5004

//
// Gstreamer1.0 compatability
//
#define GST_1_FUDGE 1       // Offset date by 1 byte as gstream has an RTP bug. Must have RTP_STREAM_SOURCE=1
