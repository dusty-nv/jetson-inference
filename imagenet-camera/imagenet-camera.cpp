/*
 * http://github.com/dusty-nv/jetson-inference
 */

#define V4L_CAMERA 0
#define GST_V4L_SRC 1

#if V4L_CAMERA
#include "v4l2Camera.h"
#else
#include "gstCamera.h"
#endif

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"
#include "cudaFont.h"
#include "imageNet.h"


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
	printf("imagenet-camera\n  args (%i):  ", argc);

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
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
#if V4L_CAMERA
	v4l2Camera* camera = v4l2Camera::Create("/dev/video0");
#else

#if GST_V4L_SRC
	int width     = 1280;
	int height    = 720;
        std::ostringstream pipeline;
	pipeline << "v4l2src device=/dev/video0 ! ";
        pipeline << "video/x-raw, width=(int)" << width << ", height=(int)" << height << ", "; 
        pipeline << "format=RGB ! ";
	pipeline << "appsink name=mysink";

        static  std::string pip = pipeline.str();

	gstCamera* camera = gstCamera::Create(pip);
#else
	gstCamera* camera = gstCamera::Create();
#endif

#endif
	
	if( !camera )
	{
		printf("\nimagenet-camera:  failed to initialize video device\n");
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
	
	if( !display ) {
		printf("\nimagenet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("imagenet-camera:  failed to create openGL texture\n");
	}
	
	
	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nimagenet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nimagenet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
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
		//else
		//	printf("imagenet-camera:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
#if GST_V4L_SRC
			printf("detectnet-camera:  failed to convert from NV12 to RGBAf\n");
#else
			printf("detectnet-camera:  failed to convert from RGB to RGBAf\n");
#endif

		// classify image
		const int img_class = net->Classify((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), &confidence);
	
		if( img_class >= 0 )
		{
			printf("imagenet-camera:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	

			if( font != NULL )
			{
				char str[256];
				sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
	
				font->RenderOverlay((float4*)imgRGBA, (float4*)imgRGBA, camera->GetWidth(), camera->GetHeight(),
								    str, 0, 0, make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}
			
			if( display != NULL )
			{
				char str[256];
				sprintf(str, "GIE build %x | %s | %s | %04.1f FPS", NV_GIE_VERSION, net->GetNetworkName(), net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);	
			}	
		}	


		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

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

			display->EndRender();
		}
	}
	
	printf("\nimagenet-camera:  un-initializing video device\n");
	
	
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
	
	printf("imagenet-camera:  video device has been un-initialized.\n");
	printf("imagenet-camera:  this concludes the test of the video device.\n");
	return 0;
}

