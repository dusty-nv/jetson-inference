/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include "imageIO.h"

#include "cudaMappedMemory.h"
#include "cudaResize.h"
#include "cudaOverlay.h"

#include "backgroundNet.h"

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
	printf("usage: backgroundnet [--help] [--replace=IMAGE] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Perform background subtraction/removal on a video or image stream.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --replace=IMAGE   perform background replacement, using this image as\n");
	printf("                    the new background. It will be resized to fit the input.\n");
	printf("  --filter-mode=MODE filtering mode used to upsample the DNN mask,\n");
	printf("                     options are:  'point' or 'linear' (default: 'linear')\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", backgroundNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

	return 0;
}

typedef uchar4 pixelType;   			// can be uchar4 or float4 (need alpha channel for background replacement)

pixelType* imgReplacement = NULL;		// replacement background image (only if --replace is used)
pixelType* imgReplacementScaled = NULL;	// replacement background image resized to the input size
pixelType* imgOutput = NULL;			// output image (background + input with alpha blending)

int2 replacementSize;
int2 replacementSizeScaled;

// replace the background of an image after it's been removed
bool replaceBackground( pixelType* imgInput, int width, int height, cudaFilterMode filter )
{
	const size_t size = width * height * sizeof(pixelType);
	
	if( !imgReplacementScaled || replacementSizeScaled.x != width || replacementSizeScaled.y != height )
	{
		CUDA_FREE(imgReplacementScaled);
		CUDA_FREE_HOST(imgOutput);

		if( !cudaAllocMapped(&imgOutput, size) )
			return false;
		
		CUDA_VERIFY(cudaMalloc(&imgReplacementScaled, size));
		CUDA_VERIFY(cudaResize(imgReplacement, replacementSize.x, replacementSize.y,
						   imgReplacementScaled, width, height, filter));
						   
		replacementSizeScaled = make_int2(width, height);
	}
	
	CUDA_VERIFY(cudaMemcpy(imgOutput, imgReplacementScaled, size, cudaMemcpyDeviceToDevice));
	CUDA_VERIFY(cudaOverlay(imgInput, width, height, imgOutput, width, height, 0, 0));
	
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
		LogError("backgroundnet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("backgroundnet:  failed to create output stream\n");	
		return 1;
	}
	

	/*
	 * create background subtraction/removal network
	 */
	backgroundNet* net = backgroundNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("backgroundnet:  failed to load backgroundNet\n");
		return 1;
	}

	// parse the desired filter mode
	const cudaFilterMode filter = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));
	
	
	/*
	 * load replacement background image if needed
	 */
	const char* replacementPath = cmdLine.GetString("replace");
	
	if( replacementPath != NULL && !loadImage(replacementPath, &imgReplacement, &replacementSize.x, &replacementSize.y) )
	{
		LogError("backgroundnet:  failed to load background replacement image %s\n", replacementPath);
		return 1;
	}
	
	
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

		// process image
		if( !net->Process(imgInput, input->GetWidth(), input->GetHeight(), filter) )
		{
			LogError("backgroundnet:  failed to process frame\n");
			continue;
		}
	
		// background replacement
		if( imgReplacement != NULL )
			replaceBackground(imgInput, input->GetWidth(), input->GetHeight(), filter);
		else
			imgOutput = imgInput;  // no bg replacement, pass through the input
		
		// render outputs
		if( output != NULL )
		{
			output->Render(imgOutput, input->GetWidth(), input->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				break;
		}
	
		// print out timing info
		net->PrintProfilerTimes();
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("backgroundnet:  shutting down...\n");
	
	CUDA_FREE_HOST(imgReplacement);
	CUDA_FREE(imgReplacementScaled);

	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	LogVerbose("backgroundnet:  shutdown complete.\n");
	return 0;
}

