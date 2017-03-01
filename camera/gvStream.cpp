/*
 * 
 */
#include "gvStream.h"
#if (VIDEO_SRC == VIDEO_GV_STREAM_SOURCE)
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"

extern void DumpHex(const void* data, size_t size);

/*
 * error - wrapper for perror
 */
void error(char *msg) {
    perror(msg);
    exit(0);
}

gvStream::gvStream(int height, int width) 
  : camera(height, width)
{
	cancel = FALSE;
	mWidth = width;
    mFrame = 0;
    mGpuBuffer = 0;
	arv_option_camera_name = NULL;
	arv_option_debug_domains = NULL;
	arv_option_snaphot = FALSE;
	arv_option_trigger = NULL;
	arv_option_software_trigger = -1;
	arv_option_frequency = -1.0;
	arv_option_width = -1;
	arv_option_height = -1;
	arv_option_horizontal_binning = -1;
	arv_option_vertical_binning = -1;
	arv_option_exposure_time_us = -1;
	arv_option_gain = -1;
	gboolean arv_option_auto_socket_buffer = FALSE;
	gboolean arv_option_no_packet_resend = FALSE;
	arv_option_packet_timeout = 20;
	arv_option_frame_retention = 100;
	arv_option_gv_stream_channel = -1;
	arv_option_gv_packet_delay = -1;
	arv_option_gv_packet_size = -1;
	gboolean arv_option_realtime = FALSE;
	gboolean arv_option_high_priority = FALSE;
	gboolean arv_option_no_packet_socket = FALSE;
	arv_option_chunks = NULL;
	arv_option_bandwidth_limit = -1;
}

bool 
gvStream::ConvertRGBtoRGBA( void* input, void** output )
{	
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, mWidth * mHeight * sizeof(float4))) )
		{
			printf(LOG_CUDA "gstCamera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
			return false;
		}
	}
	
	// GigEVision camera is configured RGB8
	if( CUDA_FAILED(cudaRGBToRGBAf((uint8_t*)input, (float4*)mRGBA, mWidth, mHeight)) )
		return false;
	
	*output = mRGBA;
	return true;
}


void
gvStream::set_cancel (int signal)
{
	cancel = TRUE;
}

void
gvStream::new_buffer_cb (ArvStream *stream, ApplicationData *data)
{
	ArvBuffer *buffer;
	size_t size = 0;

	buffer = arv_stream_try_pop_buffer (stream);
	if (buffer != NULL) {
		if (arv_buffer_get_status (buffer) == ARV_BUFFER_STATUS_SUCCESS)
			data->buffer_count++;

		if (arv_buffer_get_payload_type (buffer) == ARV_BUFFER_PAYLOAD_TYPE_CHUNK_DATA &&
		    data->chunks != NULL) {
			int i;

			for (i = 0; data->chunks[i] != NULL; i++)
				printf ("%s = %" G_GINT64_FORMAT "\n", data->chunks[i],
					arv_chunk_parser_get_integer_value (data->chunk_parser, buffer, data->chunks[i]));
		}

		/* Image processing here */
		mFrame++;
		mBuffer = (void*)arv_buffer_get_data(buffer, &size);
		printf("Frame %d, Buffer 0x%x, Size %d\n", mFrame, mBuffer, (int)size);
		DumpHex(buffer, 128);
		mFrameReady = true;
		g_main_context_wakeup(context);

		arv_stream_push_buffer (stream, buffer);
	}
}

void
gvStream::stream_cb (void *user_data, ArvStreamCallbackType type, ArvBuffer *buffer)
{
	if (type == ARV_STREAM_CALLBACK_TYPE_INIT) {
		if (arv_option_realtime) {
			if (!arv_make_thread_realtime (10))
				printf ("Failed to make stream thread realtime\n");
		} else if (arv_option_high_priority) {
			if (!arv_make_thread_high_priority (-10))
				printf ("Failed to make stream thread high priority\n");
		}
	}
}

gboolean
gvStream::periodic_task_cb (void *abstract_data)
{
	ApplicationData *data = (ApplicationData*)abstract_data;

	printf ("Frame rate = %d Hz\n", data->buffer_count);
	data->buffer_count = 0;

	if (cancel) {
		g_main_loop_quit (data->main_loop);
		return FALSE;
	}

	return TRUE;
}

gboolean
gvStream::emit_software_trigger (void *abstract_data)
{
	ArvCamera *mCamera = (ArvCamera*)abstract_data;

	arv_camera_software_trigger (mCamera);

	return TRUE;
}

void
gvStream::control_lost_cb (ArvGvDevice *gv_device)
{
	printf ("Control lost\n");

	cancel = TRUE;
}

gvStream::~gvStream(void)
{

}

char* gvStream::str_format(ArvPixelFormat format)
{
	switch (format)
	{
		case 0x01080001 :
			strcpy(tmp_str, "ARV_PIXEL_FORMAT_MONO_8");
			break;
		case 0x01080009 :
			strcpy(tmp_str, "ARV_PIXEL_FORMAT_BAYER_RG_8");
			break;
		case 0x02180014 :
			strcpy(tmp_str, "ARV_PIXEL_FORMAT_RGB_8_PACKED");
			break;
		case 0x0210001f :
			strcpy(tmp_str, "ARV_PIXEL_FORMAT_YUV_422_PACKED");
			break;
		default:
			sprintf(tmp_str, "Unknown 0x%x\n", format);
	}
	return tmp_str;
}

bool gvStream::Open()
{
#if 0
	GError *error = NULL;
	int i;

	data.buffer_count = 0;
	data.chunks = NULL;
	data.chunk_parser = NULL;

	arv_g_thread_init (NULL);
	arv_g_type_init ();

	context = g_main_context_new ();

	arv_debug_enable (arv_option_debug_domains);

	if (arv_option_camera_name == NULL)
		debug_print ("Looking for the first available camera\n");
	else
		debug_print ("Looking for camera '%s'\n", arv_option_camera_name);

	mCamera = arv_camera_new (arv_option_camera_name);
	if (mCamera != NULL) {
		gint payload;
		gint x, y, width, height;
		gint dx, dy;
		double exposure;
		guint64 n_completed_buffers;
		guint64 n_failures;
		guint64 n_underruns;
		int gain;
		software_trigger_source = 0;
		ArvPixelFormat format;
		gint maxHeight, maxWidth;

		if (arv_option_chunks != NULL) {
			char *striped_chunks;

			striped_chunks = g_strdup (arv_option_chunks);
			arv_str_strip (striped_chunks, " ,:;", ',');
			data.chunks = g_strsplit_set (striped_chunks, ",", -1);
			g_free (striped_chunks);

			data.chunk_parser = arv_camera_create_chunk_parser (mCamera);

			for (i = 0; data.chunks[i] != NULL; i++) {
				char *chunk = g_strdup_printf ("Chunk%s", data.chunks[i]);

				g_free (data.chunks[i]);
				data.chunks[i] = chunk;
			}
		}

		arv_camera_set_chunks (mCamera, arv_option_chunks);
		arv_camera_set_region (mCamera, 0, 0, arv_option_width, arv_option_height);
		arv_camera_set_binning (mCamera, arv_option_horizontal_binning, arv_option_vertical_binning);
		arv_camera_set_exposure_time (mCamera, arv_option_exposure_time_us);
		arv_camera_set_gain (mCamera, arv_option_gain);

		if (arv_camera_is_uv_device(mCamera)) {
			arv_camera_uv_set_bandwidth (mCamera, arv_option_bandwidth_limit);
		}

		if (arv_camera_is_gv_device (mCamera)) {
			arv_camera_gv_select_stream_channel (mCamera, arv_option_gv_stream_channel);
			arv_camera_gv_set_packet_delay (mCamera, arv_option_gv_packet_delay);
			arv_camera_gv_set_packet_size (mCamera, arv_option_gv_packet_size);
			arv_camera_gv_set_stream_options (mCamera, arv_option_no_packet_socket ?
							  ARV_GV_STREAM_OPTION_PACKET_SOCKET_DISABLED :
							  ARV_GV_STREAM_OPTION_NONE);
		}

		arv_camera_get_height_bounds(mCamera, NULL, &maxHeight);
		arv_camera_get_width_bounds(mCamera, NULL, &maxWidth);
		arv_camera_get_region (mCamera, &x, &y, &width, &height);
		if ((GetWidth() > width) && (GetHeight() > height))
		{
			printf("ERROR requested resolution (%d x %d) is higher than the camera can support\n", GetWidth(), GetHeight());
			return false;
		}
		x = (maxWidth / 2) - (GetWidth() / 2);
		y = (maxHeight / 2) - (GetHeight() / 2);
		width = (maxWidth / 2) + (GetWidth() / 2);
		height = (maxHeight / 2) + (GetHeight() / 2);
		
		arv_camera_set_region (mCamera, x, y, width, height);
		
		arv_camera_get_binning (mCamera, &dx, &dy);
		exposure = arv_camera_get_exposure_time (mCamera);
		payload = arv_camera_get_payload (mCamera);
		gain = arv_camera_get_gain (mCamera);
		
		/* Set your camera defaults */
		arv_camera_set_pixel_format(mCamera, VIDEO_GV_PIXEL_FORMAT);
		arv_camera_set_frame_rate(mCamera, VIDEO_GV_SRC_FRAMERATE);		

		/* Check settings */
		format = arv_camera_get_pixel_format(mCamera);
		
#if 1
		printf ("vendor name           = %s\n", arv_camera_get_vendor_name (mCamera));
		printf ("model name            = %s\n", arv_camera_get_model_name (mCamera));
		printf ("device id             = %s\n", arv_camera_get_device_id (mCamera));
		printf ("pixel format          = %s\n", str_format(format));
		if (arv_camera_is_frame_rate_available(mCamera))
		{	
			printf ("frame rate            = %f\n", arv_camera_get_frame_rate(mCamera));		
		}
		printf ("region X              = %d\n", x);
		printf ("region Y              = %d\n", y);
		printf ("image width           = %d\n", width);
		printf ("image height          = %d\n", height);
		printf ("max width             = %d\n", maxWidth);
		printf ("max height            = %d\n", maxHeight);		
		printf ("horizontal binning    = %d\n", dx);
		printf ("vertical binning      = %d\n", dy);
		printf ("payload               = %d bytes\n", payload);
		printf ("exposure              = %g µs\n", exposure);
		printf ("gain                  = %d dB\n", gain);

		if (arv_camera_is_gv_device (mCamera)) {
			printf ("gv n_stream channels  = %d\n", arv_camera_gv_get_n_stream_channels (mCamera));
			printf ("gv current channel    = %d\n", arv_camera_gv_get_current_stream_channel (mCamera));
			printf ("gv packet delay       = %" G_GINT64_FORMAT " ns\n", arv_camera_gv_get_packet_delay (mCamera));
			printf ("gv packet size        = %d bytes\n", arv_camera_gv_get_packet_size (mCamera));
		}

		if (arv_camera_is_uv_device (mCamera)) {
			guint min,max;

			arv_camera_uv_get_bandwidth_bounds (mCamera, &min, &max);
			printf ("uv bandwidth limit     = %d [%d..%d]\n", arv_camera_uv_get_bandwidth (mCamera), min, max);
		}
#endif

		mStream = arv_camera_create_stream (mCamera, stream_cb, NULL);
		if (mStream != NULL) {
			if (ARV_IS_GV_STREAM (mStream)) {
				if (arv_option_auto_socket_buffer)
					g_object_set (mStream,
						      "socket-buffer", ARV_GV_STREAM_SOCKET_BUFFER_AUTO,
						      "socket-buffer-size", 0,
						      NULL);
				if (arv_option_no_packet_resend)
					g_object_set (mStream,
						      "packet-resend", ARV_GV_STREAM_PACKET_RESEND_NEVER,
						      NULL);
				g_object_set (mStream,
					      "packet-timeout", (unsigned) arv_option_packet_timeout * 1000,
					      "frame-retention", (unsigned) arv_option_frame_retention * 1000,
					      NULL);
			}

			for (i = 0; i < 50; i++)
				arv_stream_push_buffer (mStream, arv_buffer_new (payload, NULL));

			arv_camera_set_acquisition_mode (mCamera, ARV_ACQUISITION_MODE_CONTINUOUS);

			if (arv_option_frequency > 0.0)
				arv_camera_set_frame_rate (mCamera, arv_option_frequency);

			if (arv_option_trigger != NULL)
				arv_camera_set_trigger (mCamera, arv_option_trigger);

			if (arv_option_software_trigger > 0.0) {
				arv_camera_set_trigger (mCamera, "Software");
				software_trigger_source = g_timeout_add ((double) (0.5 + 1000.0 /
										   arv_option_software_trigger),
									 emit_software_trigger, mCamera);
			}

			arv_camera_start_acquisition (mCamera);

			g_signal_connect (mStream, "new-buffer", G_CALLBACK (new_buffer_cb), &data);
			arv_stream_set_emit_signals (mStream, TRUE);

			g_signal_connect (arv_camera_get_device (mCamera), "control-lost",
					  G_CALLBACK (control_lost_cb), NULL);

			g_timeout_add_seconds (1, periodic_task_cb, &data);

			data.main_loop = g_main_loop_new (NULL, FALSE);

			old_sigint_handler = signal (SIGINT, set_cancel);

//			g_main_loop_run (data.main_loop);


		} else
			printf ("Can't create stream thread (check if the device is not already used)\n");

	} else
	{
		printf ("No camera found\n");
		return false;
	}

#endif
	return true;
}

void gvStream::Close()
{
#if 0
	arv_stream_get_statistics (mStream, &n_completed_buffers, &n_failures, &n_underruns);

	printf ("Completed buffers = %Lu\n", (unsigned long long) n_completed_buffers);
	printf ("Failures          = %Lu\n", (unsigned long long) n_failures);
	printf ("Underruns         = %Lu\n", (unsigned long long) n_underruns);
#endif
	g_object_unref (mCamera);

	if (software_trigger_source > 0)
		g_source_remove (software_trigger_source);

	signal (SIGINT, old_sigint_handler);

	g_main_loop_unref (data.main_loop);
	
	arv_camera_stop_acquisition (mCamera);

	g_object_unref (mStream);

	if (data.chunks != NULL)
		g_strfreev (data.chunks);

	g_clear_object (&data.chunk_parser);
}

bool gvStream::Capture( void** cpu, void** cuda, unsigned long timeout )
{
	printf ("DEBUG1: set buffer 0x%x\n", mGpuBuffer);
	// Allocate a buffer the first time we call this function
	if (!mGpuBuffer) cudaMalloc(&mGpuBuffer, GetHeight() * GetWidth() * 3);
	printf ("DEBUG: set buffer 0x%x\n", mGpuBuffer);
	
#if 0	
	mFrameReady = false;
	if (!cancel)
	{
		printf("0 (%d x %d = %d)\n", GetWidth(), GetHeight(), GetWidth() * GetHeight() * 3);
		while (!mFrameReady)
		{
			printf("1\n");
			g_main_context_iteration(context, TRUE);
		}
		mFrameReady = false;		
		printf("2\n");
	}
#endif

//	cudaMemcpy( mGpuBuffer, mBuffer, GetWidth() * GetHeight() * 3, cudaMemcpyHostToDevice );
	if (cudaMemset( mGpuBuffer, 0xaa, GetWidth() * GetHeight() * 3) != cudaSuccess)
		printf ("ERROR: could not set buffer 0x%x\n", mGpuBuffer);;
	
//bufferIn = (char*)malloc(height * width * 2); // Holds YUV data

	*cpu = (void*)mBuffer;
	*cuda = (void*)gpuBuffer;
	return true;
}

#endif


