/*
 * http://github.com/ross-abaco/jetson-inference
 *
 */

#include <string>

#if V4L_CAMERA
#include "v4l2Camera.h"
#else
#include "rtpStream.h"
#include "gstCamera.h"
#include "gvStream.h"
#endif

#include "config.h"
#include "debug.h"

#if SDL_DISPLAY
#include "sdlDisplay.h"
#include "glTexture.h"
#define glDisplay sdlDisplay
#else
#include "glDisplay.h"
#include "glTexture.h"
#endif

#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <stdio.h>

#include "cudaNormalize.h"
#include "cudaYUV.h"
#include "imageNet.h"

using namespace std;

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		debug_print("received SIGINT\n");
		signal_recieved = true;
	}
}

bool display_confidence = false;

void key_handler(char key)
{
    switch (key)
    {
        case ' ' :
            display_confidence ? display_confidence=false : display_confidence=true;
            break;
        default:
            printf("unregistered keypress (%c), ignoring\n", key);
    }
}

void* mRGBA = 0;

bool ConvertYUVtoRGBf( uint8_t* input, void** output, uint32_t width, uint32_t height )
{
	if( !input || !output )
		return false;

	if( !mRGBA )
	{
		if( CUDA_FAILED(cudaMalloc(&mRGBA, width * height * sizeof(float4))) )
		{
			printf(LOG_CUDA "cudaMalloc -- failed to allocate memory for %ux%u RGBA texture\n", width, height);
			return false;
		}
	}

	// nvcamera is YUV
	if( CUDA_FAILED(cudaYUVToRGBAf((uint8_t*)input, (float4*)mRGBA, width, height)) )
		{
			printf(LOG_CUDA "cudaYUVToRGBAf -- failed convert %ux%u RGBA texture\n", width, height);
			return false;
		}

	*output = mRGBA;

	return true;
}

int main( int argc, char** argv )
{
    char tmp[200][200];
    float tmpConf[200];

    unsigned int frame = 0;
    int itemCount = 0;
    char str[256];

#if ABACO
    int logo;
#endif
    SDL_Color white = {255, 255, 255, 0}; // WWhite
    SDL_Color orange = {247, 107, 34, 0}; // Abaco orange
    SDL_Color black = {40, 40, 40, 0}; // Black

	debug_print("imagenet-camera\n  args (%i):  ", argc);

	// Some help.
	printf("Keyboard commands:\n\tF     Fullscreen\n\tQ     Quit\n\tSPACE Toggle overlay");

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");

	/*
	 * parse network type from CLI arguments
	 */
	imageNet::NetworkType networkType = imageNet::GOOGLENET;

	if( argc > 1 && strcmp(argv[1], "alexnet") == 0 )
		networkType = imageNet::ALEXNET;

	if( signal(SIGINT, sig_handler) == SIG_ERR )
		debug_print("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	 */

#if VIDEO_SRC==VIDEO_GST_RTP_SRC || VIDEO_SRC==VIDEO_GST_V4L_SRC
	int width     = WIDTH;
	int height    = HEIGHT;
    std::ostringstream pipeline;
#endif

#if VIDEO_SRC==VIDEO_GST_RTP_SRC
	pipeline << "udpsrc address=239.192.1.44 port=5004 caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)" << width << ", height=(string)" << height << ", payload=(int)96\" ! ";
	pipeline << "rtpvrawdepay ! queue ! ";
	pipeline << "appsink name=mysink";
#endif

#if VIDEO_SRC==VIDEO_GST_V4L_SRC
	pipeline << "v4l2src device=" << VIDEO_GST_V4L_SRC_DEVICE << " ! ";
    pipeline << "video/x-raw, width=(int)" << width << ", height=(int)" << height << ", framerate=" << VIDEO_GST_V4L_SRC_FRAMERATE << "/1, ";
    pipeline << "format=RGB ! ";
	pipeline << "appsink name=mysink";
#endif

#if VIDEO_SRC==VIDEO_GST_RTP_SRC || VIDEO_SRC==VIDEO_GST_V4L_SRC
    static  std::string pip = pipeline.str();
    std::cout << pip << "\n";
	gstCamera* camera = gstCamera::Create(pip, HEIGHT, WIDTH);
#endif

#if VIDEO_SRC==VIDEO_RTP_STREAM_SOURCE
	rtpStream* camera = new rtpStream(HEIGHT, WIDTH);
	camera->rtpStreamIn((char*)IP_UNICAST, IP_PORT_IN);
#endif

#if VIDEO_SRC==VIDEO_GV_STREAM_SOURCE
	gvStream* camera = new gvStream(HEIGHT, WIDTH);
#endif

#if VIDEO_SRC==VIDEO_NV // USB Nvida TX1 CSI Webcam
	gstCamera* camera = gstCamera::Create();
#endif

	if( !camera )
	{
		debug_print("\nimagenet-camera:  failed to initialize video device\n");
		return 0;
	}

	printf("\nimagenet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());

	/*
	 * create imageNet
	 */
	imageNet* net = imageNet::Create(networkType);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create(camera->GetWidth(), camera->GetHeight());
	glTexture* texture = NULL;
#if GST_RTP_SINK
	rtpStream rtpStreaming(HEIGHT, WIDTH, (char*)"127.0.0.1", 5004);
	rtpStreaming.Open();
#endif

	/*
	 * register keyboard callback
	 */
	display->RegisterKeyCallback(key_handler);

	if( !display ) {
		printf("\nimagenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);
        texture->Render(display->mRenderer);
		if( !texture )
			printf("imagenet-camera:  failed to create openGL texture\n");
	}


#if ABACO
	/*
	 * load logo
	 */
    logo = texture->ImageLoad((char*)"abaco.bmp");
#endif

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nimagenet-camera:  failed to open camera for streaming\n");
		return 0;
	}

	debug_print("\nimagenet-camera:  camera open for streaming\n");


	/*
	 * processing loop
	 */
	float confidence = 0.0f;

	while( ( !signal_recieved) && (!display->Quit() ) )
	{
		void* dummy  = NULL;
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		// convert from YUV to RGBA
		void* imgRGBA = NULL;

		// get the latest frame

printf("Captureding\n");
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\nimagenet-camera:  failed to capture frame\n");
printf("Captured\n");
#if VIDEO_SRC==VIDEO_GST_RTP_SRC
		if ( !camera->ConvertYUVtoRGBA(imgCUDA, &imgRGBA) )
			printf("imagenet-camera:  failed to convert from YUV to RGBAf\n");
#endif
#if VIDEO_SRC==VIDEO_RTP_STREAM_SOURCE
		if ( !ConvertYUVtoRGBf((uint8_t*)imgCUDA, &imgRGBA, camera->GetWidth(), camera->GetHeight() ) )
			printf("imagenet-camera:  failed to convert from YUV to RGBAf\n");
#endif
#if VIDEO_SRC==VIDEO_GST_V4L_SRC || VIDEO_GV_STREAM_SOURCE
		if ( !camera->ConvertRGBtoRGBA(imgCUDA, &imgRGBA) )
			printf("imagenet-camera:  failed to convert from RGB to RGBAf\n");
printf("Converted\n");
#endif
#if VIDEO_SRC==VIDEO_NV
		if ( !camera->ConvertNV12toRGBA(imgCUDA, &imgRGBA) )
			printf("imagenet-camera:  failed to convert from NV12 to RGBAf\n");
#endif
printf("Classifying %dx%d\n", camera->GetWidth(), camera->GetHeight());
		// classify image
		const int img_class = net->Classify((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);

		if( img_class >= 0 )
		{
			debug_print("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));

			sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));

			if( display != NULL )
			{
				char banner[256];
				sprintf(banner, "TensorRT build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net->GetNetworkName(), net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				display->SetTitle(banner);
			}
		}
		else
		{
			printf("imagenet-camera:  classify failure, aborting\n");
			return 0;
		}

		// update display
		if( display != NULL )
		{
			int baroffset=0;
			display->UserEvents();
			display->BeginRender();

#if ABACO
            texture->Image(320, 0, logo);
            texture->Box(0, camera->GetHeight()-40, camera->GetWidth(), camera->GetHeight(), 0xF16B22FF);
#endif
			if (camera->GetHeight() < 720)
			{
				texture->Box(0, 0, camera->GetWidth(), 35, 0xF16B22FF);
				baroffset = 62;
			}
			else
			{
				texture->Box(0, 0, camera->GetWidth(), 52, 0xF16B22FF);
				baroffset = 103;
			}


            // Update confidence/s less frequently
            if (frame++ % 50 == 0)
            {
                // Bubble sort the list
                {
                    bool swap = true;
                    while (swap)
                    {
                        swap = false;
                        for (int ii=0;ii<net->mItems.count-1;ii++)
                        {
                            if (net->mItems.index[ii].confidence < net->mItems.index[ii+1].confidence)
                            {

                                float tmpf;
                                unsigned int tmpn;

                                tmpf = net->mItems.index[ii].confidence;
                                net->mItems.index[ii].confidence = net->mItems.index[ii+1].confidence;
                                net->mItems.index[ii+1].confidence = tmpf;

                                tmpn = net->mItems.index[ii].number;
                                net->mItems.index[ii].number = net->mItems.index[ii+1].number;
                                net->mItems.index[ii+1].number = tmpn;

                                swap=true;
                            }
                        }
                    }
                }

                for (itemCount=0;itemCount<net->mItems.count;itemCount++)
                {
                    sprintf((char*)&tmp[itemCount][0], "%05.2f%% %s", net->mItems.index[itemCount].confidence * 100.0f, net->GetClassDesc(net->mItems.index[itemCount].number));
                    tmpConf[itemCount] = net->mItems.index[itemCount].confidence;
                }
            }

			// limit displayed items to 15
			if (itemCount > 15 )
				itemCount = 15;

            // Render the confidence bar
            if (display_confidence)
            {
                for (int ii=0;ii<itemCount;ii++)
                {
                  int t;
                  long c = 0;

                  // First bar is green rest are red
                  if (ii == 0)
                    c = 0x2BB24CFF;
                  else
                    c = 0xD34536FF;

                  t = tmpConf[ii] * 100.0f;
                  texture->Box(45, baroffset+(ii*22), 45 + ((300 / 100) * t), baroffset+20+(ii*22), c);
                }
            }

			// Render the video
  			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f),
								   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(0,0);
			}

            if (display_confidence)
            {
                // Render the text
                for (int ii=0;ii<itemCount;ii++)
                {
                  texture->RenderText((char*)&tmp[ii][0], white, 50, baroffset-3+(ii*22), 18);
                }
            }

			if (camera->GetHeight() < 720)
			{
				texture->RenderText(str, white, 15, 5, 18);
			}
			else
			{
				texture->RenderText(str, white, 15, 5, 28);
			}

#if ABACO
			if (camera->GetWidth() < 1024)
			{
				texture->RenderText((char*)"abaco.com", white, 15, camera->GetHeight()-32, 18);
			}
			else
			{
				texture->RenderText((char*)"WE INNOVATE. WE DELIVER. ", black, camera->GetWidth()-381, camera->GetHeight()-33, 18);
				texture->RenderText((char*)"YOU SUCCEED.", white, camera->GetWidth()-140, camera->GetHeight()-33, 18);
				texture->RenderText((char*)"abaco.com", white, 15, camera->GetHeight()-40, 28);
			}
#endif

#if GST_RTP_SINK
			rtpStreaming.Transmit((char*)imgCPU);
#endif
		    display->EndRender();
		}
	}

	debug_print("\nimagenet-camera:  un-initializing video device\n");

    if( net != NULL )
	{
		delete net;
		net = NULL;
	}

	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}

#if GST_RTP_SINK
	rtpStreaming.Close();
#endif

	debug_print("imagenet-camera:  video device has been un-initialized.\n");
	debug_print("imagenet-camera:  this concludes the test of the video device.\n");\
    printf("Done./n");
	return 0;
}

