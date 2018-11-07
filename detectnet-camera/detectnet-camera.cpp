/*
 * http://github.com/ross-nv/jetson-inference
 */
#include "config.h"

#if V4L_CAMERA
#include "v4l2Camera.h"
#else
#include "gstCamera.h"
#include "rtpStream.h"
#include "gvStream.h"
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

void convertColour(camera *camera, void* imgCUDA, void** imgRGBA)
{
#if VIDEO_SRC==VIDEO_GST_RTP_SRC
	if ( !camera->ConvertYUVtoRGBA(imgCUDA, imgRGBA) )
		printf("imagenet-camera:  failed to convert from YUV to RGBAf\n");
#endif

#if VIDEO_SRC==VIDEO_RTP_STREAM_SOURCE
	if ( !camera->ConvertYUVtoRGBf(imgCUDA, imgRGBA ) )
		printf("imagenet-camera:  failed to convert from YUV to RGBAf\n");
#endif

#if VIDEO_SRC==VIDEO_GST_V4L_SRC
	if ( !camera->ConvertRGBtoRGBA(imgCUDA, imgRGBA) )
		printf("imagenet-camera:  failed to convert from RGB to RGBAf\n");
#endif 

#if  VIDEO_GV_STREAM_SOURCE
#if VIDEO_GV_PIXEL_FORMAT == VIDEO_GV_RGB8
	if ( !camera->ConvertRGBtoRGBA(imgCUDA, imgRGBA) )
		printf("imagenet-camera:  failed to convert from RGB to RGBAf\n");
#elif VIDEO_GV_PIXEL_FORMAT == VIDEO_GV_YUV422
	if ( !camera->ConvertYUVtoRGBf(imgCUDA, imgRGBA) )
		printf("imagenet-camera:  failed to convert from YUV to RGBAf\n");
#elif VIDEO_GV_PIXEL_FORMAT == VIDEO_GV_BAYER_GR8
	if ( !camera->ConvertBAYER_GR8toRGBA(imgCUDA, imgRGBA) )
		printf("imagenet-camera:  failed to convert from BAYER_GR8 to RGBAf\n");
#endif
#endif
}

int main( int argc, char** argv )
{
#if LOGO
    int logo;

    SDL_Color white = {255, 255, 255, 0}; // White
//    SDL_Color orange = {247, 107, 34, 0}; // Abaco orange
    SDL_Color orange = {136, 201, 70, 0}; // Green orange #88c946
    SDL_Color black = {40, 40, 40, 0}; // Black
#endif

#if LOGO
	printf("%s (%s) Inferance Demonstration\n", COMPANY_NAME, COMPANY_URL);
#else
	printf("Jetson Inferance Demonstration\n");
#endif
	printf("\tAuthor : ross@rossnewman.com, dustinf@nvidia.com\n");
	printf("\tVideo Src : %s\n", VIDEO_SRC_NAME);
	printf("\tBytes Per Pixel : %u%s\n",VIDEO_BYTES_PER_PIXEL, VIDEO_GV_PIXEL_FORMAT_NAME);
	printf("\tHidth : %u\n", HEIGHT);
	printf("\tHeight : %u\n", WIDTH);
	printf("\tFramerate : %f\n\n", VIDEO_DEFAULT_FRAMERATE);
	
	/*
	 * parse network type from CLI arguments
	 */
	detectNet::NetworkType networkType = detectNet::PEDNET_MULTI;

	networkType = DETECTNET_TYPE;
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	 */
#if (VIDEO_SRC==VIDEO_GST_V4L_SRC) || (VIDEO_SRC==GST_RTP_SRC)
	int width     = WIDTH;
	int height    = HEIGHT;
    std::ostringstream pipeline;
#if VIDEO_SRC==VIDEO_GST_RTP_SRC
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
#elif VIDEO_SRC==VIDEO_GV_STREAM_SOURCE
	gvStream* camera = new gvStream(HEIGHT, WIDTH);
#else
	gstCamera* camera = gstCamera::Create();
#endif
	
	if( !camera )
	{
		printf("\ndetectnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized video device\n");
	
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
	
#if LOGO
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
		void* imgRGBA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\ndetectnet-camera:  failed to capture frame\n");
		
		/*
		 *  Convert capture colorspace to the required RGBA
		 */
		convertColour(camera, imgCUDA, &imgRGBA);

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
				sprintf(str, "TensorRT build %u.u.u | %s | %04.1f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();
			
#if LOGO
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
#if LOGO
			if (!(camera->GetWidth() < 1024))
			{
				texture->RenderText((char*)"WE INNOVATE. WE DELIVER. ", black, camera->GetWidth()-381, camera->GetHeight()-33, 18); 
				texture->RenderText((char*)"YOU SUCCEED.", white, camera->GetWidth()-140, camera->GetHeight()-33, 18); 
				texture->RenderText((char*)"abaco.com", white, 15, camera->GetHeight()-40, 28); 
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

