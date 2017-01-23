/* Example RTP packet from wireshark
      Real-Time Transport Protocol
      10.. .... = Version: RFC 1889 Version (2)
      ..0. .... = Padding: False
      ...0 .... = Extension: False
      .... 0000 = Contributing source identifiers count: 0
      0... .... = Marker: False
      Payload type: DynamicRTP-Type-96 (96)
      Sequence number: 34513
      Timestamp: 2999318601
      Synchronization Source identifier: 0xdccae7a8 (3704285096)
      Payload: 000003c000a08000019e00a2000029292929f06e29292929...
*/

/*

Gstreamer1.0 working example UYVY streaming
===========================================
gst-launch-1.0 videotestsrc num_buffers ! video/x-raw, format=UYVY, framerate=25/1, width=640, height=480 ! queue ! rtpvrawpay ! udpsink host=127.0.0.1 port=5004

gst-launch-1.0 udpsrc port=5004 caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)480, height=(string)480, payload=(int)96" ! queue ! rtpvrawdepay ! queue ! xvimagesink sync=false


Use his program to stream data to the udpsc example above on the tegra X1

*/

#ifndef __RTP_STREAM_H__
#define __RTP_STREAM_H__

#include <byteswap.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include "camera.h"


#define ARM                   1    			/* Perform endian swap */
#define RTP_VERSION           0x2  			/* RFC 1889 Version 2 */
#define RTP_PADDING           0x0
#define RTP_EXTENSION         0x0
#define RTP_MARKER            0x0
#define RTP_PAYLOAD_TYPE      0x60 			/* 96 Dynamic Type */
#define RTP_SOURCE            0x12345678 	/* Sould be unique */
#define RTP_FRAMERATE         25

#define Hz90                  90000
#define NUM_LINES_PER_PACKET  1 			/* can have more that one line in a packet */
#define MAX_BUFSIZE 	      1280 * 3 		/* allow for RGB data upto 1280 pixels wide */
#define MAX_UDP_DATA 		  1500  		/* enough space for three lines of UDP data MTU size should be checked */

static unsigned long sequence_number;

/* 12 byte RTP Raw video header */
typedef struct
{
  int32_t protocol;
  int32_t timestamp;
  int32_t source;
} rtp_header;


typedef struct  __attribute__((__packed__))
{
  int16_t length;
  int16_t line_number;
  int16_t offset;
} line_header;

typedef struct __attribute__((__packed__))
{
  int16_t extended_sequence_number;
  line_header line[NUM_LINES_PER_PACKET];
} payload_header;


typedef struct  __attribute__((__packed__))
{
  rtp_header rtp;
  payload_header payload;
} header;

typedef struct
{
  header head;
  char data[MAX_BUFSIZE];
} rtp_packet;


/**
 * rtpstream RGB data
 */
class rtpStream : public camera
{
public:
    rtpStream(int height, int width);
    ~rtpStream();
	void rtpStreamOut(char* hostname, int port);
	void rtpStreamIn(char* hostname, int port);
	int Transmit(char* rgbframe, bool gpuAddr);
    bool Open();
	void Close();
    bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX );
	void update_header(header *packet, int line, int last, int32_t timestamp, int32_t source);
    int mSockfdIn;
    int mSockfdOut;
    struct sockaddr_in mServeraddrIn;
    struct sockaddr_in mServeraddrOut;
    socklen_t mServerlenIn;
    socklen_t mServerlenOut;
    pthread_mutex_t mutex;
    unsigned int mFrame;
  	char* gpuBuffer;
	char udpdata[MAX_UDP_DATA];
    char* bufferIn;
private:
    struct hostent *mServerIn;
    struct hostent *mServerOut;
    int mWidth;
	// Ingress port
    char mHostnameIn[100];
    int mPortNoIn;
	// Egress port
    char mHostnameOut[100];
    int mPortNoOut;
};

typedef struct
{
	char* rgbframe;
	char* yuvframe;
	bool gpuAddr;
	uint32_t width;
	uint32_t height;
	rtpStream *stream;
} tx_data;

#if ARM
	void endianswap32(uint32_t *data, int length);
	void endianswap16(uint16_t *data, int length);
#endif

#endif
