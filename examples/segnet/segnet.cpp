/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "videoSource.h"
#include "videoOutput.h"

#include "cudaOverlay.h"
#include "cudaMappedMemory.h"

#include "segNet.h"

#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: segnet [--help] [--network NETWORK] ...\n");
	printf("              input_URI [output_URI]\n\n");
	printf("Segment and classify a video/image stream using a semantic segmentation DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s\n", segNet::Usage());
	printf("%s\n", videoSource::Usage());
	printf("%s\n", videoOutput::Usage());
	printf("%s\n", Log::Usage());

	return 0;
}


//
// segmentation buffers
//
typedef uchar3 pixelType;		// this can be uchar3, uchar4, float3, float4

pixelType* imgMask      = NULL;	// color of each segmentation class
pixelType* imgOverlay   = NULL;	// input + alpha-blended mask
pixelType* imgComposite = NULL;	// overlay with mask next to it
pixelType* imgOutput    = NULL;	// reference to one of the above three

int2 maskSize;
int2 overlaySize;
int2 compositeSize;
int2 outputSize;

// allocate mask/overlay output buffers
bool allocBuffers( int width, int height, uint32_t flags )
{
	// check if the buffers were already allocated for this size
	if( imgOverlay != NULL && width == overlaySize.x && height == overlaySize.y )
		return true;

	// free previous buffers if they exit
	CUDA_FREE_HOST(imgMask);
	CUDA_FREE_HOST(imgOverlay);
	CUDA_FREE_HOST(imgComposite);

	// allocate overlay image
	overlaySize = make_int2(width, height);
	
	if( flags & segNet::VISUALIZE_OVERLAY )
	{
		if( !cudaAllocMapped(&imgOverlay, overlaySize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for overlay image (%ux%u)\n", width, height);
			return false;
		}

		imgOutput = imgOverlay;
		outputSize = overlaySize;
	}

	// allocate mask image (half the size, unless it's the only output)
	if( flags & segNet::VISUALIZE_MASK )
	{
		maskSize = (flags & segNet::VISUALIZE_OVERLAY) ? make_int2(width/2, height/2) : overlaySize;

		if( !cudaAllocMapped(&imgMask, maskSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for mask image\n");
			return false;
		}

		imgOutput = imgMask;
		outputSize = maskSize;
	}

	// allocate composite image if both overlay and mask are used
	if( (flags & segNet::VISUALIZE_OVERLAY) && (flags & segNet::VISUALIZE_MASK) )
	{
		compositeSize = make_int2(overlaySize.x + maskSize.x, overlaySize.y);

		if( !cudaAllocMapped(&imgComposite, compositeSize) )
		{
			LogError("segnet:  failed to allocate CUDA memory for composite image\n");
			return false;
		}

		imgOutput = imgComposite;
		outputSize = compositeSize;
	}

	return true;
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
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("segnet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("segnet:  failed to create output stream\n");	
		return 1;
	}
	

	/*
	 * create segmentation network
	 */
	segNet* net = segNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("segnet:  failed to initialize segNet\n");
		return 1;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", 150.0f));

	// get the desired overlay/mask filtering mode
	const segNet::FilterMode filterMode = segNet::FilterModeFromStr(cmdLine.GetString("filter-mode", "linear"));

	// get the visualization flags
	const uint32_t visualizationFlags = segNet::VisualizationFlagsFromStr(cmdLine.GetString("visualize", "overlay|mask"));

	// get the object class to ignore (if any)
	const char* ignoreClass = cmdLine.GetString("ignore-class", "void");

	
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image
		pixelType* imgInput = NULL;
		int status = 0;
		
		if( !input->Capture(&imgInput, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}

		// allocate buffers for this size frame
		if( !allocBuffers(input->GetWidth(), input->GetHeight(), visualizationFlags) )
		{
			LogError("segnet:  failed to allocate buffers\n");
			continue;
		}

		// process the segmentation network
		if( !net->Process(imgInput, input->GetWidth(), input->GetHeight(), ignoreClass) )
		{
			LogError("segnet:  failed to process segmentation\n");
			continue;
		}
		
		// generate overlay
		if( visualizationFlags & segNet::VISUALIZE_OVERLAY )
		{
			if( !net->Overlay(imgOverlay, overlaySize.x, overlaySize.y, filterMode) )
			{
				LogError("segnet:  failed to process segmentation overlay.\n");
				continue;
			}
		}

		// generate mask
		if( visualizationFlags & segNet::VISUALIZE_MASK )
		{
			if( !net->Mask(imgMask, maskSize.x, maskSize.y, filterMode) )
			{
				LogError("segnet:-console:  failed to process segmentation mask.\n");
				continue;
			}
		}

		// generate composite
		if( (visualizationFlags & segNet::VISUALIZE_OVERLAY) && (visualizationFlags & segNet::VISUALIZE_MASK) )
		{
			CUDA(cudaOverlay(imgOverlay, overlaySize, imgComposite, compositeSize, 0, 0));
			CUDA(cudaOverlay(imgMask, maskSize, imgComposite, compositeSize, overlaySize.x, 0));
		}

		// render outputs
		if( output != NULL )
		{
			output->Render(imgOutput, outputSize.x, outputSize.y);

			// update the status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);

			// check if the user quit
			if( !output->IsStreaming() )
				break;
		}

		// wait for the GPU to finish		
		CUDA(cudaDeviceSynchronize());

		// print out timing info
		net->PrintProfilerTimes();
	}
	

	/*
	 * destroy resources
	 */
	LogVerbose("segnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);

	CUDA_FREE_HOST(imgMask);
	CUDA_FREE_HOST(imgOverlay);
	CUDA_FREE_HOST(imgComposite);

	LogVerbose("segnet:  shutdown complete.\n");
	return 0;
}

