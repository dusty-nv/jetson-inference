/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "depthNet.h"

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
	printf("usage: depthnet [--help] [--network NETWORK]\n");
	printf("                [--colormap COLORMAP] [--filter-mode MODE]\n");
	printf("                [--visualize VISUAL] [--depth-size SIZE]\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Mono depth estimation on a video/image stream using depthNet DNN.\n\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network=NETWORK pre-trained model to load (see below for options)\n");
	printf("  --visualize=VISUAL controls what is displayed (e.g. --visualize=input,depth)\n");
	printf("                     valid combinations are:  'input', 'depth' (comma-separated)\n");
	printf("  --depth-size=SIZE  scales the size of the depth map visualization, as a\n");
	printf("                     percentage of the input size (default is 1.0)\n");
	printf("  --filter-mode=MODE filtering mode used during visualization,\n");
	printf("                     options are:  'point' or 'linear' (default: 'linear')\n");
	printf("  --colormap=COLORMAP depth colormap (default is 'viridis-inverted')\n");
	printf("                      options are:  'inferno', 'inferno-inverted',\n");
	printf("                                    'magma', 'magma-inverted',\n");
	printf("                                    'parula', 'parula-inverted',\n");
	printf("                                    'plasma', 'plasma-inverted',\n");
	printf("                                    'turbo', 'turbo-inverted',\n");
	printf("                                    'viridis', 'viridis-inverted'\n\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", depthNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());
	
	return 0;
}


//
// depth map buffers
//
typedef uchar3 pixelType;        // this can be uchar3, uchar4, float3, float4

pixelType* imgDepth = NULL;      // colorized depth map image
pixelType* imgComposite = NULL;  // original image with depth map next to it

int2 inputSize;
int2 depthSize;
int2 compositeSize;

// allocate depth map & output buffers
bool allocBuffers( int width, int height, uint32_t flags, float depthScale )
{
	// check if the buffers were already allocated for this size
	if( imgDepth != NULL && width == inputSize.x && height == inputSize.y )
		return true;

	// free previous buffers if they exit
	CUDA_FREE_HOST(imgDepth);
	CUDA_FREE_HOST(imgComposite);

	// allocate depth map
	inputSize = make_int2(width, height);
	depthSize = make_int2(width * depthScale, height * depthScale);
	
	if( !cudaAllocMapped(&imgDepth, depthSize) )
	{
		LogError("depthnet:  failed to allocate CUDA memory for depth map (%ix%i)\n", depthSize.x, depthSize.y);
		return false;
	}

	// allocate composite image
	compositeSize = make_int2(0,0);
	
	if( flags & depthNet::VISUALIZE_DEPTH )
	{
		compositeSize.x += depthSize.x;
		compositeSize.y = depthSize.y;
	}
	
	if( flags & depthNet::VISUALIZE_INPUT )
	{
		compositeSize.x += inputSize.x;
		compositeSize.y = inputSize.y;
	}
	
	if( !cudaAllocMapped(&imgComposite, compositeSize) )
	{
		LogError("depthnet:  failed to allocate CUDA memory for composite image (%ix%i)\n", compositeSize.x, compositeSize.y);
		return false;
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
		LogError("depthnet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("depthnet:  failed to create output stream\n");
		return 1;
	}
	

	/*
	 * create mono-depth network
	 */
	depthNet* net = depthNet::Create(cmdLine);

	if( !net )
	{
		LogError("depthnet:   failed to initialize depthNet\n");
		return 1;
	}

	// parse the desired colormap
	const cudaColormapType colormap = cudaColormapFromStr(cmdLine.GetString("colormap", "viridis-inverted"));

	// parse the desired filter mode
	const cudaFilterMode filterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));

	// parse the visualization flags
	const uint32_t visualizationFlags = depthNet::VisualizationFlagsFromStr(cmdLine.GetString("visualize"));
	
	// get the depth map size scaling factor
	const float depthScale = cmdLine.GetFloat("depth-size", 1.0);


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
		if( !allocBuffers(input->GetWidth(), input->GetHeight(), visualizationFlags, depthScale) )
		{
			LogError("depthnet:  failed to allocate output buffers\n");
			continue;
		}
		
		// infer the depth and visualize the depth map
		if( !net->Process(imgInput, inputSize.x, inputSize.y, 
					   imgDepth, depthSize.x, depthSize.y, 
					   colormap, filterMode) )
		{
			LogError("depthnet-camera:  failed to process depth map\n");
			continue;
		}

		// overlay the images into composite output image
		if( visualizationFlags & depthNet::VISUALIZE_INPUT )
			CUDA(cudaOverlay(imgInput, inputSize, imgComposite, compositeSize, 0, 0));
		
		if( visualizationFlags & depthNet::VISUALIZE_DEPTH )
			CUDA(cudaOverlay(imgDepth, depthSize, imgComposite, compositeSize, (visualizationFlags & depthNet::VISUALIZE_INPUT) ? inputSize.x : 0, 0));
		
		// render outputs
		if( output != NULL )
		{
			output->Render(imgComposite, compositeSize.x, compositeSize.y);

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
	LogVerbose("depthnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	CUDA_FREE_HOST(imgDepth);
	CUDA_FREE_HOST(imgComposite);

	LogVerbose("depthnet:  shutdown complete.\n");
	return 0;
}

