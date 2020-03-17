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

#include "cudaWarp.h"
#include "cudaMappedMemory.h"
#include "commandLine.h"

#include "homographyNet.h"
#include "mat33.h"

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
	 * setup exit signal handler
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
		printf("\nhomography-camera:  failed to initialize camera device\n");
		return 0;
	}
	
	const uint32_t imgWidth  = camera->GetWidth();
	const uint32_t imgHeight = camera->GetHeight();

	printf("\nhomography-camera:  successfully initialized camera device\n");
	printf("    width:  %u\n", imgWidth);
	printf("   height:  %u\n", imgHeight);
	printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());
	

	/*
	 * create homography network
	 */
	homographyNet* net = homographyNet::Create(argc, argv);
	
	if( !net )
	{
		printf("homography-camera:   failed to initialize homographyNet\n");
		return 0;
	}


	/*
	 * allocate memory for warped image
	 */
	float4* imgWarpedCPU  = NULL;
	float4* imgWarpedCUDA = NULL;

	if( !cudaAllocMapped((void**)&imgWarpedCPU, (void**)&imgWarpedCUDA, imgWidth * imgHeight * sizeof(float4)) )
	{
		printf("homography-console:  failed to allocate CUDA memory for warped image\n");
		return 0;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("homography-camera:  failed to create openGL display\n");
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("homography-camera:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("homography-camera:  camera open for streaming\n");
	
	
	/*
	 * stabilize the camera video
	 */
	float* lastImg = NULL;

	float displacementAvg[] = {0,0,0,0,0,0,0,0};	// average the camera displacement over a series of frames 
	const float displacementAvgFactor = 1.0f;	// to smooth it out over time (factor of 1.0 = instant)
	
	while( !signal_recieved )
	{
		// capture RGBA image
		float* imgRGBA = NULL;

		if( !camera->CaptureRGBA(&imgRGBA, 1000) )
			printf("homography-camera:  failed to capture frame\n");

		// make sure we have 2 frames to use
		if( !lastImg )
		{
			lastImg = imgRGBA;
			continue;
		}

		// find the displacement
		float displacement[8];

		if( !net->FindDisplacement(lastImg, imgRGBA, imgWidth, imgHeight, displacement) )
		{
			printf("homography-camera:  failed to find displacement\n");
			continue;
		}

		// smooth the displacement
		for( uint32_t n=0; n < 8; n++ )
			displacementAvg[n] = displacement[n] * displacementAvgFactor + displacementAvg[n] * (1.0f - displacementAvgFactor);

		// find the homography
		float H[3][3];
		float H_inv[3][3];

		if( !net->ComputeHomography(displacementAvg, H, H_inv) )
		{
			printf("homography-camera:  failed to find homography\n");
			continue;
		}

		mat33_print(H, "H");
		mat33_print(H_inv, "H_inv");

		// stabilize the latest frame by warping it by H_inverse to align with the previous frame
		if( CUDA_FAILED(cudaWarpPerspective((float4*)imgRGBA, imgWarpedCUDA, imgWidth, imgHeight, H_inv, false)) )
		{
			printf("homography-console:  failed to warp output image\n");
			continue;
		}

		// update display
		if( display != NULL )
		{
			// render the image
			display->RenderOnce((float*)imgWarpedCUDA, imgWidth, imgHeight);

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS | Display %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS(), display->GetFPS());
			display->SetTitle(str);

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;	
		}

		lastImg = imgRGBA;
	}
	
	
	/*
	 * destroy resources
	 */
	printf("homography-camera:  shutting down...\n");
	
	SAFE_DELETE(camera);
	SAFE_DELETE(display);
	SAFE_DELETE(net);

	printf("homography-camera:  shutdown complete.\n");
	return 0;
}

