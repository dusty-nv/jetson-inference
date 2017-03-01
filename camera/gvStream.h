/* Example GigEVison packet from wireshark


*/
#include "config.h"
#if (VIDEO_SRC == VIDEO_GV_STREAM_SOURCE)
#ifndef __GV_STREAM_H__
#define __GV_STREAM_H__

#include <arv.h>
#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include "camera.h"

typedef struct {
	GMainLoop *main_loop;
	int buffer_count;
	ArvChunkParser *chunk_parser;
	char **chunks;
} ApplicationData;

static gboolean cancel;
static char *arv_option_camera_name;
static char *arv_option_debug_domains;
static gboolean arv_option_snaphot;
static char *arv_option_trigger;
static double arv_option_software_trigger;
static double arv_option_frequency;
static int arv_option_width;
static int arv_option_height;
static int arv_option_horizontal_binning;
static int arv_option_vertical_binning;
static double arv_option_exposure_time_us;
static int arv_option_gain;
static gboolean arv_option_auto_socket_buffer;
static gboolean arv_option_no_packet_resend;
static unsigned int arv_option_packet_timeout;
static unsigned int arv_option_frame_retention;
static int arv_option_gv_stream_channel;
static int arv_option_gv_packet_delay;
static int arv_option_gv_packet_size;
static gboolean arv_option_realtime;
static gboolean arv_option_high_priority;
static gboolean arv_option_no_packet_socket;
static char *arv_option_chunks;
static unsigned int arv_option_bandwidth_limit;
		
static unsigned int mFrame;
static bool mFrameReady;
static GMainContext *context;
static const void *mBuffer;

/**
 * gvStream ethernet video data (Aravis)
 */
class gvStream : public camera
{
public:
    gvStream(int height, int width);
    ~gvStream();
    int Transmit(char* rgbframe, bool gpuAddr) {return -1;}; /* Not supported */
    bool Open();
	void Close();
    bool Capture( void** cpu, void** cuda, unsigned long timeout=ULONG_MAX );
    bool ConvertRGBtoRGBA( void* input, void** output );
	/* GigE Vision functions */
	static void new_buffer_cb (ArvStream *stream, ApplicationData *data);
	static void stream_cb (void *user_data, ArvStreamCallbackType type, ArvBuffer *buffer);
	static gboolean periodic_task_cb (void *abstract_data);
	static gboolean emit_software_trigger (void *abstract_data);
	static void control_lost_cb (ArvGvDevice *gv_device);
	static void set_cancel (int signal);
private:
    char *str_format(ArvPixelFormat format);
	ApplicationData data;
	ArvCamera *mCamera;
	ArvStream *mStream;
	guint software_trigger_source;
	void (*old_sigint_handler)(int);
	char *mGpuBuffer;
	char tmp_str[200];
		
	void* mRGBA;
};

#endif
#endif
