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

#include "commandLine.h"
#include "cudaMappedMemory.h"

#include "segNet.h"

#include <signal.h>


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
	commandLine cmdLine(argc, argv);
	
	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
								   cmdLine.GetInt("height", gstCamera::DefaultHeight),
								   cmdLine.GetString("camera"));

	if( !camera )
	{
		printf("\nsegnet-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	printf("\nsegnet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create segmentation network
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
	
	if( !display )
		printf("segnet-camera:  failed to create openGL display\n");
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("segnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("segnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		// capture RGBA image
		float* imgRGBA = NULL;
		
		if( !camera->CaptureRGBA(&imgRGBA, 1000, true) )
			printf("segnet-camera:  failed to convert from NV12 to RGBA\n");

		// process the segmentation network
		if( !net->Process(imgRGBA, camera->GetWidth(), camera->GetHeight()) )
		{
			printf("segnet-console:  failed to process segmentation\n");
			continue;
		}

		// generate overlay
		if( !net->Overlay(outCUDA, camera->GetWidth(), camera->GetHeight(), segNet::FILTER_LINEAR) )
		{
			printf("segnet-console:  failed to process segmentation overlay.\n");
			continue;
		}
		
		// update display
		if( display != NULL )
		{
			// render the image
			display->RenderOnce((float*)imgRGBA, camera->GetWidth(), camera->GetHeight());

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS | Display %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), 1000.0f / net->GetNetworkTime(), display->GetFPS());
			display->SetTitle(str);

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}
	}
	

	/*
	 * destroy resources
	 */
	printf("segnet-camera:  shutting down...\n");
	
	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("segnet-camera:  shutdown complete.\n");
	return 0;
}

