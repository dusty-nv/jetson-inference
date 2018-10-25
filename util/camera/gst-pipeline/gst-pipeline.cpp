/*
 * inference-101
 */

#include "gstPipeline.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"


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
	printf("gst-pipeline\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
		
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the pipeline
	 */
	gstPipeline* pipeline = gstPipeline::Create(
		"rtspsrc location=rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mov ! queue ! rtph264depay ! h264parse ! queue ! omxh264dec ! appsink name=mysink",
		240,
		160,
		12
	);
	
	if( !pipeline )
	{
		printf("\ngst-pipeline:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\ngst-pipeline:  successfully initialized video device\n");
	printf("    width:  %u\n", pipeline->GetWidth());
	printf("   height:  %u\n", pipeline->GetHeight());
	printf("    depth:  %u (bpp)\n", pipeline->GetPixelDepth());
	


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("\ngst-pipeline:  failed to create openGL display\n");

	const size_t texSz = pipeline->GetWidth() * pipeline->GetHeight() * sizeof(float4);
	float4* texIn = (float4*)malloc(texSz);

	/*if( texIn != NULL )
		memset(texIn, 0, texSz);*/

	if( texIn != NULL )
		for( uint32_t y=0; y < pipeline->GetHeight(); y++ )
			for( uint32_t x=0; x < pipeline->GetWidth(); x++ )
				texIn[y*pipeline->GetWidth()+x] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);

	glTexture* texture = glTexture::Create(pipeline->GetWidth(), pipeline->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/, texIn);

	if( !texture )
		printf("gst-pipeline:  failed to create openGL texture\n");
	
	

	/*
	 * start streaming
	 */
	if( !pipeline->Open() )
	{
		printf("\ngst-pipeline:  failed to open pipeline for streaming\n");
		return 0;
	}
	
	printf("\ngst-pipeline:  pipeline open for streaming\n");
	
	
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !pipeline->Capture(&imgCPU, &imgCUDA, 10000) )
			printf("\ngst-pipeline:  failed to capture frame\n");
		else
			printf("gst-pipeline:  recieved new frame  CPU=0x%p  GPU=0x%p\n", imgCPU, imgCUDA);
		
		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !pipeline->ConvertRGBA(imgCUDA, &imgRGBA) )
			printf("gst-pipeline:  failed to convert from NV12 to RGBA\n");

		// rescale image pixel intensities
		CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
						   (float4*)imgRGBA, make_float2(0.0f, 1.0f),
                               pipeline->GetWidth(), pipeline->GetHeight()));

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					CUDA(cudaDeviceSynchronize());

					texture->Unmap();
				}
				//texture->UploadCPU(texIn);

				texture->Render(100,100);		
			}

			display->EndRender();
		}
	}
	
	printf("\ngst-pipeline:  un-initializing video device\n");
	
	
	/*
	 * shutdown the pipeline device
	 */
	if( pipeline != NULL )
	{
		delete pipeline;
        pipeline = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("gst-pipeline:  video device has been un-initialized.\n");
	printf("gst-pipeline:  this concludes the test of the video device.\n");
	return 0;
}
