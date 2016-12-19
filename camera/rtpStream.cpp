#include "rtpStream.h"

typedef struct float4 {
    float x;
    float y;
    float z;
    float w;
} float4;

/* 
 * error - wrapper for perror
 */
void error(char *msg) {
    perror(msg);
    exit(0);
}

void rgbtoyuv(int y, int x, char* yuv, char* rgb)
{
  int c,cc,R,G,B,Y,U,V;
  int size;
  
  cc=0;
  size = x*3;
  for (c=0;c<size;c+=3)
  {
    R=rgb[c];
    G=rgb[c+1];
    B=rgb[c+2];
    /* sample luma for every pixel */
    Y  =      (0.257 * R) + (0.504 * G) + (0.098 * B) + 16;
    yuv[cc]=Y;
    if (c % 2 == 0)
    {
        V =  (0.439 * R) - (0.368 * G) - (0.071 * B) + 128;
        yuv[cc+1]=V;
    }
    else
    {
        U = -(0.148 * R) - (0.291 * G) + (0.439 * B) + 128;
        yuv[cc+1]=U;
    }
    cc+=2;
  }
}

/* Broadcast the stream to port 5004 */
rtpStream::rtpStream(int height, int width, char* hostname, int portno) :
    camera(height, width)
{
	strcpy(mHostname, hostname);
    mPortNo = portno;
    mFrame = 0;
}

bool rtpStream::Open() 
{
    /* socket: create the socket */
    mSockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (mSockfd < 0) 
    {
        printf("ERROR opening socket");
	    return error;
	}

    /* gethostbyname: get the server's DNS entry */
    mServer = gethostbyname(mHostname);
    if (mServer == NULL) {
        fprintf(stderr,"ERROR, no such host as %s\n", mHostname);
        exit(0);
    }

    /* build the server's Internet address */
    bzero((char *) &mServeraddr, sizeof(mServeraddr));
    mServeraddr.sin_family = AF_INET;
    bcopy((char *)mServer->h_addr, 
	  (char *)&mServeraddr.sin_addr.s_addr, mServer->h_length);
    mServeraddr.sin_port = htons(mPortNo);

    /* send the message to the server */
    mServerlen = sizeof(mServeraddr);
    
    return true;
}

void rtpStream::Close() 
{
	close(mSockfd);
}

#if ARM
void rtpStream::endianswap32(uint32_t *data, int length)
{
  int c = 0;
  for (c=0;c<length;c++)
    data[c] = __bswap_32 (data[c]);
}

void rtpStream::endianswap16(uint16_t *data, int length)
{
  int c = 0;
  for (c=0;c<length;c++)
    data[c] = __bswap_16 (data[c]);
}
#endif

void rtpStream::update_header(header *packet, int line, int last, int32_t timestamp, int32_t source)
{
  bzero((char *)packet, sizeof(header));
  packet->rtp.protocol = RTP_VERSION << 30;
  packet->rtp.protocol = packet->rtp.protocol | RTP_PAYLOAD_TYPE << 16;
  packet->rtp.protocol = packet->rtp.protocol | sequence_number++;
  /* leaving other fields as zero TODO Fix*/
  packet->rtp.timestamp = timestamp += (Hz90 / RTP_FRAMERATE);
  packet->rtp.source = source;
  packet->payload.extended_sequence_number = 0; /* TODO : Fix extended seq numbers */
  packet->payload.line[0].length = MAX_BUFSIZE;
  packet->payload.line[0].line_number = line;
  packet->payload.line[0].offset = 0;
  if (last==1)
  {
    packet->rtp.protocol = packet->rtp.protocol | 1 << 23;
  } 
#if 0
  printf("0x%x, 0x%x, 0x%x \n", packet->rtp.protocol, packet->rtp.timestamp, packet->rtp.source);
  printf("0x%x, 0x%x, 0x%x \n", packet->payload.line[0].length, packet->payload.line[0].line_number, packet->payload.line[0].offset);
#endif
}

int rtpStream::Transmit(char* rgbframe) 
{
    rtp_packet packet;
    char *yuv;
    int c=0;
    int n=0;
    

    sequence_number=0;
    
#if 0
#if 0  
    float4 *RGBAf = (float4*)rgbframe;
    
    // RGBAf to RGB
    for( int y=0; y < 100; y++ )
	{
		for( int x=0; x < mWidth; x++ )
		{
			rgbframe[(y*(mWidth*3))+(x*3)] = (char)RGBAf[(y*mWidth)+x].x;
			rgbframe[(y*(mWidth*3))+(x*3)+1] = (char)RGBAf[(y*mWidth)+x].y;
			rgbframe[(y*(mWidth*3))+(x*3)+2] = (char)RGBAf[(y*mWidth)+x].z;
		}
	}
#else
    for( int y=0; y < mHeight; y++ )
	{
		for( int x=0; x < mWidth; x+=3 )
		{
			rgbframe[(y*(mWidth*3))+(x*3)] = (char)0xff;
			rgbframe[(y*(mWidth*3))+(x*3)+1] = (char)0;
			rgbframe[(y*(mWidth*3))+(x*3)+2] = (char)0;
		}
	}
#endif
#endif

    
    /* get a message from the user */
    bzero(packet.data, MAX_BUFSIZE);
    
    /* send a frame */
    {
      struct timeval NTP_value;
      int32_t time = 10000;
      
      for (c=0;c<(mHeight);c++)
      {
        int x,last = 0;
        if (c==mHeight-1) last=1;
        update_header((header*)&packet, c, last, time, RTP_SOURCE);
        x = c * (mWidth * 3);
        
#if 1
        rgbtoyuv(mHeight, mWidth, packet.data, (char*)&rgbframe[x]);
#else
        memcpy((char*)packet.data, (char*)&rgbframe[x], mWidth* 3);
#endif

#if ARM
        endianswap32((uint32_t *)&packet, sizeof(rtp_header)/4);
        endianswap16((uint16_t *)&packet.head.payload, sizeof(payload_header)/2);
#endif
        n = sendto(mSockfd, (char *)&packet, sizeof(rtp_packet), 0, (const sockaddr*)&mServeraddr, mServerlen);
        if (n < 0) 
          fprintf(stderr, "ERROR in sendto");
      }
      
      printf("Sent frame %d\n", mFrame++);
    }
        
    return 0;
}

