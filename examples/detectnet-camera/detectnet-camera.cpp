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

#include "detectNet.h"
#include "commandLine.h"

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
	printf("usage: detectnet-camera [-h] [--network NETWORK] [--camera CAMERA]\n");
	printf("                        [--width WIDTH] [--height HEIGHT]\n\n");
	printf("Locate objects in a live camera stream using an object detection DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help           show this help message and exit\n");
	printf("  --camera CAMERA  index of the MIPI CSI camera to use (NULL for CSI camera 0),\n");
	printf("                   or for VL42 cameras the /dev/video node to use (/dev/video0).\n");
     printf("                   by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH    desired width of camera stream (default is 1280 pixels)\n");
	printf("  --height HEIGHT  desired height of camera stream (default is 720 pixels)\n\n");
	printf("%s\n", detectNet::Usage());

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
		printf("\ndetectnet-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	printf("\ndetectnet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create detection network
	 */
	detectNet* net = detectNet::Create(argc, argv);
	
	if( !net )
	{
		printf("detectnet-camera:   failed to load detectNet model\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();

	if( !display ) 
		printf("detectnet-camera:  failed to create openGL display\n");


	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("detectnet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("detectnet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	float confidence = 0.0f;
	
	while( !signal_recieved )
	{
		// capture RGBA image
		float* imgRGBA = NULL;
		
		if( !camera->CaptureRGBA(&imgRGBA, 1000) )
			printf("detectnet-camera:  failed to capture RGBA image from camera\n");

		// detect objects in the frame
		detectNet::Detection* detections = NULL;
	
		const int numDetections = net->Detect(imgRGBA, camera->GetWidth(), camera->GetHeight(), &detections);
		
		if( numDetections > 0 )
		{
			printf("%i objects detected\n", numDetections);
		
			for( int n=0; n < numDetections; n++ )
			{
				printf("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID, net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
				printf("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left, detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(), detections[n].Height()); 
			}
		}	

		// update display
		if( display != NULL )
		{
			// render the image
			display->RenderOnce(imgRGBA, camera->GetWidth(), camera->GetHeight());

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), 1000.0f / net->GetNetworkTime());
			display->SetTitle(str);

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	

	/*
	 * destroy resources
	 */
	printf("detectnet-camera:  shutting down...\n");
	
	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("detectnet-camera:  shutdown complete.\n");
	return 0;
}

