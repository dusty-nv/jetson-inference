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

int usage()
{
	printf("usage: segnet-camera [-h] [--network NETWORK] [--camera CAMERA]\n");
	printf("                     [--width WIDTH] [--height HEIGHT]\n");
	printf("                     [--alpha ALPHA] [--filter-mode MODE]\n");
	printf("                     [--ignore-class CLASS]\n\n");
	printf("Segment and classify a live camera stream using a semantic segmentation DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network NETWORK pre-trained model to load (see below for options)\n");
	printf("  --camera CAMERA   index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("                    or for VL42 cameras the /dev/video device to use.\n");
     printf("                    by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH     desired width of camera stream (default: 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of camera stream (default: 720 pixels)\n");
	printf("  --alpha ALPHA     overlay alpha blending value, range 0-255 (default: 120)\n");
	printf("  --filter-mode MODE   filtering mode used during visualization,\n");
	printf("                       options are 'point' or 'linear' (default: 'linear')\n");
	printf("  --ignore-class CLASS optional name of class to ignore when classifying\n");
	printf("                       the visualization results (default: 'void')\n\n");
	printf("%s\n", segNet::Usage());

	return 0;
}

int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();

	
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
	
	const uint32_t width = camera->GetWidth();
	const uint32_t height = camera->GetHeight();

	printf("\nsegnet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", width);
	printf("   height:  %u\n", height);
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
	net->SetGlobalAlpha(cmdLine.GetFloat("alpha", 120.0f));

	// get the desired alpha blend filtering mode
	const segNet::FilterMode filterMode = segNet::FilterModeFromStr(cmdLine.GetString("filter-mode", "linear"));

	// get the object class to ignore (if any)
	const char* ignoreClass = cmdLine.GetString("ignore-class", "void");

	
	/*
	 * allocate segmentation overlay output buffers
 	 */
	float* imgOverlay = NULL;
	float* imgMask    = NULL;

	if( !cudaAllocMapped((void**)&imgOverlay, width * height * sizeof(float) * 4) )
	{
		printf("segnet-camera:  failed to allocate CUDA memory for overlay image (%ux%u)\n", width, height);
		return 0;
	}

	if( !cudaAllocMapped((void**)&imgMask, width/2 * height/2 * sizeof(float) * 4) )
	{
		printf("segnet-camera:  failed to allocate CUDA memory for mask image\n");
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
		if( !net->Process(imgRGBA, width, height, ignoreClass) )
		{
			printf("segnet-console:  failed to process segmentation\n");
			continue;
		}
		
		// generate overlay
		if( !net->Overlay(imgOverlay, width, height, filterMode) )
		{
			printf("segnet-console:  failed to process segmentation overlay.\n");
			continue;
		}

		// generate mask
		if( !net->Mask(imgMask, width/2, height/2, filterMode) )
		{
			printf("segnet-console:  failed to process segmentation mask.\n");
			continue;
		}
		
		// update display
		if( display != NULL )
		{
			// begin the frame
			display->BeginRender();

			// render the images
			display->Render(imgOverlay, width, height);
			display->Render(imgMask, width/2, height/2, width);

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
			display->SetTitle(str);

			// present the frame
			display->EndRender();

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}

		// wait for the GPU to finish		
		CUDA(cudaDeviceSynchronize());

		// print out timing info
		net->PrintProfilerTimes();
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

