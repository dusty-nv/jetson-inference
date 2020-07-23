/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include "cudaCrop.h"
#include "cudaWarp.h"
#include "cudaMappedMemory.h"

#include "commandLine.h"
#include "stereoNet.h"

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
	printf("usage: stereonet-camera [-h] [--network NETWORK] [--camera CAMERA]\n");
	printf("                        [--width WIDTH] [--height HEIGHT]\n");
	printf("                        [--colormap COLORMAP] [--filter-mode MODE]\n");
	printf("Stereo disparity depth estimation on a live camera stream using DNN.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network NETWORK pre-trained model to load (see below for options)\n");
	printf("  --camera CAMERA   index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("                    or for VL42 cameras the /dev/video device to use.\n");
     printf("                    by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH     desired width of camera stream (default: 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of camera stream (default: 720 pixels)\n");
	printf("  --colormap COLORMAP  colormap to use (default is 'viridis')\n");
	printf("  --filter-mode MODE   filtering mode used during visualization,\n");
	printf("                       options are 'point' or 'linear' (default: 'linear')\n\n");
	printf("%s\n", stereoNet::Usage());

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
		printf("\nstereonet-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	const uint32_t cameraWidth = camera->GetWidth();
	const uint32_t cameraHeight = camera->GetHeight();

	const uint32_t width = cameraWidth / 2;	 // stereo camera has both L/R
	const uint32_t height = cameraHeight;	 // frames included in one image

	printf("\nstereonet-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", cameraWidth);
	printf("   height:  %u\n", cameraHeight);
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create mono-depth network
	 */
	stereoNet* net = stereoNet::Create(stereoNet::RESNET18_2D); //(argc, argv);

	if( !net )
	{
		printf("stereonet-camera:   failed to initialize stereoNet\n");
		return 0;
	}

	// parse the desired colormap
	const cudaColormapType colormap = cudaColormapFromStr(cmdLine.GetString("colormap"));

	// parse the desired filter mode
	const cudaFilterMode filterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));

	
	/*
	 * allocate left/right inputs
	 */
	float* imgInput[] = { NULL, NULL, NULL, NULL };

	for( int n=0; n < 4; n++ )
	{
		if( CUDA_FAILED(cudaMalloc((void**)&imgInput[n], width * height * sizeof(float) * 4)) )
		{
			printf("stereonet-camera:  failed to allocate CUDA memory for input image #%i (%ix%i)\n", n, width, height);
			return 0;
		}
	}

	/*
	 * allocate output depth map
	 */
	float* imgDepth = NULL;

	if( CUDA_FAILED(cudaMalloc((void**)&imgDepth, width * height * sizeof(float) * 4)) )
	{
		printf("stereonet-camera:  failed to allocate CUDA memory for output image (%ix%i)\n", width, height);
		return 0;
	}
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("stereonet-camera:  failed to create openGL display\n");
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("stereonet-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("stereonet-camera:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture camera image
		float* imgCamera = NULL;
		
		if( !camera->CaptureRGBA(&imgCamera, 1000) )
			printf("stereonet-camera:  failed to capture image\n");

		// apply left/right image cropping
		if( CUDA_FAILED(cudaCropRGBA((float4*)imgCamera, (float4*)imgInput[0],
							    make_int4(0, 0, width, height), 
							    cameraWidth, cameraHeight)) )
		{
			printf("stereonet-camera:  failed to crop left input image\n");
			continue;
		}

		if( CUDA_FAILED(cudaCropRGBA((float4*)imgCamera, (float4*)imgInput[1],
							    make_int4(width, 0, cameraWidth, cameraHeight), 
							    cameraWidth, cameraHeight)) )
		{
			printf("stereonet-camera:  failed to crop right input image\n");
			continue;
		}

#if 0
		// apply intrinsic calibration
		if( CUDA_FAILED(cudaWarpIntrinsic((float4*)imgInput[0], (float4*)imgInput[2], width, height, 
									make_float2(349.938f, 349.938f), make_float2(341.8f, 181.698f),
									make_float4(-0.17182f, 0.026142f, 0.0f, 0.0f))) )
		{
			printf("stereonet-camera:  failed to de-warp left input image\n");
			continue;
		}

		if( CUDA_FAILED(cudaWarpIntrinsic((float4*)imgInput[1], (float4*)imgInput[3], width, height, 
									make_float2(349.99f, 349.99f), make_float2(327.78f, 193.343f),
									make_float4(-0.17067f, 0.025132f, 0.0f, 0.0f))) )
		{
			printf("stereonet-camera:  failed to de-warp right input image\n");
			continue;
		}
#endif

		// process the depth mapping
		if( !net->Process(imgInput[0], imgInput[1], imgDepth, 
					   width, height, colormap, filterMode) )
		{
			printf("stereonet-camera:  failed to process depth map\n");
			continue;
		}
		
		// update display
		if( display != NULL )
		{
			// begin the frame
			display->BeginRender();

			// render the images
			display->Render(imgInput[0], width, height);
			display->Render(imgInput[1], width, height, width);
			display->Render(imgDepth, width, height, 0, height+30);
			//display->Render(imgDepth, width, height, width);

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
	printf("stereonet-camera:  shutting down...\n");
	
	CUDA(cudaFree(imgInput[0]));
	CUDA(cudaFree(imgInput[1]));
	CUDA(cudaFree(imgDepth));

	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("stereonet-camera:  shutdown complete.\n");
	return 0;
}

