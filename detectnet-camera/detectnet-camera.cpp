/*
 * http://github.com/dusty-nv/jetson-inference
 */

#define V4L_CAMERA 0
#define GST_V4L_SRC 1
#define GST_RTP_SRC 0
#define SDL_DISPLAY 1
#define ABACO 1

#if GST_RTP_SRC
#define HEIGHT 480
#define WIDTH 640
#else
#define HEIGHT 720
#define WIDTH 1280
#endif

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

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"

#include "detectNet.h"


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
#if ABACO
    int logo;

    SDL_Color white = {255, 255, 255, 0}; // WWhite
    SDL_Color orange = {247, 107, 34, 0}; // Abaco orange
    SDL_Color black = {40, 40, 40, 0}; // Black
#endif

	printf("detectnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	

	/*
	 * parse network type from CLI arguments
	 */
	detectNet::NetworkType networkType = detectNet::PEDNET_MULTI;

	if( argc > 1 )
	{
		if( strcmp(argv[1], "ped-100") == 0 )
			networkType = detectNet::PEDNET;
		else if( strcmp(argv[1], "facenet-120") == 0 || strcmp(argv[1], "face-120") == 0 )
			networkType = detectNet::FACENET;
	}
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
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

	gstCamera* camera = gstCamera::Create(pip, height, width);
#else
	gstCamera* camera = gstCamera::Create();
#endif
	
	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detectNet
	 */
	detectNet* net = detectNet::Create(networkType);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * allocate memory for output bounding boxes and class confidence
	 */
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create(camera->GetWidth(), camera->GetHeight());
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ndetectnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("detectnet-camera:  failed to create openGL texture\n");
	}
	
	/*
	 * load logo
	 */
    logo = texture->ImageLoad("abaco.bmp");
    
	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ndetectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !display->Quit() && !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ndetectnet-camera:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
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

		// classify image with detectNet
		int numBoundingBoxes = maxBoxes;
	
		if( net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU))
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);
		
			int lastClass = 0;
			int lastStart = 0;
			
			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);
				
				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
				
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), 
						                        bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-console:  failed to draw boxes\n");
						
					lastClass = nc;
					lastStart = n;
				}
			}
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();
			
#if ABACO
            texture->Image(320, 0, logo);
            texture->Box(0, camera->GetHeight()-40, camera->GetWidth(), camera->GetHeight(), 0xF16B22FF);
#endif

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
#if ABACO
			if (!(camera->GetWidth() < 1024))
			{
				texture->RenderText("WE INNOVATE. WE DELIVER. ", black, camera->GetWidth()-381, camera->GetHeight()-33, 18); 
				texture->RenderText("YOU SUCCEED.", white, camera->GetWidth()-140, camera->GetHeight()-33, 18); 
				texture->RenderText("abaco.com", white, 15, camera->GetHeight()-40, 28); 
			}
#endif
			display->EndRender();
		}
	}
	
	printf("\ndetectnet-camera:  un-initializing video device\n");
	
	
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
	
	printf("detectnet-camera:  video device has been un-initialized.\n");
	printf("detectnet-camera:  this concludes the test of the video device.\n");
	return 0;
}

