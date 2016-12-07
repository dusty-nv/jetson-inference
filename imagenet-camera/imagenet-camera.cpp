/*
 * http://github.com/ross-abaco/jetson-inference
 *
 * sudo apt-get install libsdl2-dev
 */

#define V4L_CAMERA 0
#define GST_V4L_SRC 0
#define GST_RTP_SRC 1
#define SDL_DISPLAY 1
#define ABACO 1

#if 0
#define HEIGHT 720
#define WIDTH 1280
#else
#define HEIGHT 480
#define WIDTH 640
#endif

#include "debug.h"
#include <string>

#if V4L_CAMERA
#include "v4l2Camera.h"
#else
#include "gstCamera.h"
#endif

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
#include "cudaFont.h"
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

int main( int argc, char** argv )
{
    char tmp[200][200];
    float tmpConf[200];

    unsigned int frame = 0;
    int itemCount = 0;

#if ABACO
    char str[256];
    int logo;

    SDL_Color white = {255, 255, 255, 0}; // WWhite
    SDL_Color orange = {247, 107, 34, 0}; // Abaco orange
    SDL_Color black = {40, 40, 40, 0}; // Black
#endif

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
#if V4L_CAMERA
	v4l2Camera* camera = v4l2Camera::Create("/dev/video0");
#else

#if GST_V4L_SRC || GST_RTP_SRC
	int width     = WIDTH;
	int height    = HEIGHT;
    std::ostringstream pipeline;
#if GST_RTP_SRC
	pipeline << "udpsrc address=239.192.1.44 port=5004 caps=\"application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)RAW, sampling=(string)YCbCr-4:2:2, depth=(string)8, width=(string)" << width << ", height=(string)" << height << ", payload=(int)96\" ! ";
	pipeline << "queue ! rtpvrawdepay ! queue ! ";
	pipeline << "appsink name=mysink";
#else
	pipeline << "v4l2src device=/dev/video0 ! ";
    pipeline << "video/x-raw, width=(int)" << width << ", height=(int)" << height << ", "; 
    pipeline << "format=RGB ! ";
	pipeline << "appsink name=mysink";
#endif
    static  std::string pip = pipeline.str();

	gstCamera* camera = gstCamera::Create(pip);
#else
	gstCamera* camera = gstCamera::Create();
#endif

#endif
	
	if( !camera )
	{
		debug_print("\nimagenet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	debug_print("\nimagenet-camera:  successfully initialized video device\n");
	debug_print("    width:  %u\n", camera->GetWidth());
	debug_print("   height:  %u\n", camera->GetHeight());
	debug_print("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

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
		
	/*
	 * create font
	 */
//	cudaFont* font = cudaFont::Create();
	
#if ABACO
	/*
	 * load logo
	 */
    logo = texture->ImageLoad("abaco.bmp");
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
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		// get the latest frame
#if V4L_CAMERA
        cudaMalloc(&imgRGBA,  camera->GetWidth() *  camera->GetHeight() * sizeof(float4));

        imgCPU = camera->Capture();
		if (imgCPU)
            memcpy(imgRGBA, camera->Capture(), camera->GetWidth() *  camera->GetHeight() * sizeof(float4));
        else
            printf("imagenet-camera: V4l2 did not get buffer 0x%x!!\n", imgCPU);
#else
		if( !camera->Capture(&imgCUDA, &imgCPU, 1000) )
			printf("\nimagenet-camera:  failed to capture frame\n");
#endif
	
#if GST_V4L_SRC || GST_RTP_SRC
#if GST_V4L_SRC 
		if( !camera->ConvertRGBtoRGBA(imgCUDA, &imgRGBA) )
			printf("detectnet-camera:  failed to convert from RGB to RGBAf\n");
#endif
#if GST_RTP_SRC
		if( !camera->ConvertYUVtoRGBA(imgCUDA, &imgRGBA) )
			printf("detectnet-camera:  failed to convert from YUV to RGBAf\n");
#endif
#else
		if( !camera->ConvertNV12toRGBA(imgCUDA, &imgRGBA) )
			printf("detectnet-camera:  failed to convert from NV12 to RGBAf\n");
#endif

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
				texture->Box(0, 0, camera->GetWidth(), 65, 0xF16B22FF);
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

#if ABACO
			if (camera->GetHeight() < 720)
			{
				texture->RenderText(str, white, 15, 5, 18); 
			}
			else
			{
				texture->RenderText(str, white, 15, 5, 28); 
			}
			
			if (camera->GetWidth() < 1024)
			{
				texture->RenderText("abaco.com", white, 15, camera->GetHeight()-30, 18); 
			}
			else
			{
				texture->RenderText("WE INNOVATE. WE DELIVER. ", black, camera->GetWidth()-381, camera->GetHeight()-33, 18); 
				texture->RenderText("YOU SUCCEED.", white, camera->GetWidth()-140, camera->GetHeight()-33, 18); 
				texture->RenderText("abaco.com", white, 15, camera->GetHeight()-40, 28); 
			}
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
	
	debug_print("imagenet-camera:  video device has been un-initialized.\n");
	debug_print("imagenet-camera:  this concludes the test of the video device.\n");\
    printf("Done./n");
	return 0;
}

