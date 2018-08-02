/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "segNet.h"


#define DEFAULT_CAMERA -1	// -1 for onboard camera, or change to index of /dev/video V4L2 camera (>=0)	
		

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
	printf("segnet-camera\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\nsegnet-camera:  failed to initialize video device\n");
		return 0;
	}
	
	printf("\nsegnet-camera:  successfully initialized video device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create segNet
	 */
	segNet* net = segNet::Create(argc, argv);
	
	if( !net )
	{
		printf("segnet-camera:   failed to initialize imageNet\n");
		return 0;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetGlobalAlpha(120);

	// allocate segmentation overlay output buffer
	float* outCPU  = NULL;
	float* outCUDA = NULL;

	if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, camera->GetWidth() * camera->GetHeight() * sizeof(float) * 4) )
	{
		printf("segnet-camera:  failed to allocate CUDA memory for output image (%ux%u)\n", camera->GetWidth(), camera->GetHeight());
		return 0;
	}

	
	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\nsegnet-camera:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("segnet-camera:  failed to create openGL texture\n");
	}
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nsegnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\nsegnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// get the latest frame
		if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
			printf("\nsegnet-camera:  failed to capture frame\n");

		// convert from YUV to RGBA
		void* imgRGBA = NULL;
		
		if( !camera->ConvertRGBA(imgCUDA, &imgRGBA, true) )
			printf("segnet-camera:  failed to convert from NV12 to RGBA\n");

		// process image overlay
		if( !net->Overlay((float*)imgRGBA, (float*)outCUDA, camera->GetWidth(), camera->GetHeight()) )
		{
			printf("segnet-console:  failed to process segmentation overlay.\n");
			continue;
		}
		
		// update display
		if( display != NULL )
		{
			char str[256];
			sprintf(str, "TensorRT build %i.%i.%i | %s | %04.1f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
			display->SetTitle(str);	
	
			// next frame in the window
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)outCUDA, make_float2(0.0f, 255.0f), 
								   (float4*)outCUDA, make_float2(0.0f, 1.0f), 
		 						   camera->GetWidth(), camera->GetHeight()));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, outCUDA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(10, 10);		
			}

			display->EndRender();
		}
	}
	
	printf("\nsegnet-camera:  un-initializing video device\n");
	
	
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
	
	printf("segnet-camera:  video device has been un-initialized.\n");
	printf("segnet-camera:  this concludes the use of the device.\n");
	return 0;
}

