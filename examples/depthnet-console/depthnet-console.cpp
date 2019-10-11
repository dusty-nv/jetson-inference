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

#include "depthNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"


int usage()
{
	printf("usage: depthnet-console [-h] [--network NETWORK]\n");
	printf("                        [--colormap COLORMAP] [--filter-mode MODE]\n");
	printf("                        file_in file_out\n\n");
	printf("Mono depth estimation from an image using depthNet DNN.\n\n");
	printf("positional arguments:\n");
	printf("  file_in              filename of the input image to process\n");
	printf("  file_out             filename of the output image to save\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
	printf("  --colormap COLORMAP  colormap to use (default is 'viridis')\n");
	printf("  --filter-mode MODE   filtering mode used during visualization,\n");
	printf("                       options are 'point' or 'linear' (default: 'linear')\n\n");
	printf("%s\n", depthNet::Usage());

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
	 * parse filename arguments
	 */
	const char* imgFilename = cmdLine.GetPosition(0);
	const char* outFilename = cmdLine.GetPosition(1);

	if( !imgFilename || !outFilename )
	{
		printf("depthnet-console:   input and output image filenames required\n");
		return usage();
	}


	/*
	 * create mono-depth network
	 */
	depthNet* net = depthNet::Create(argc, argv);

	if( !net )
	{
		printf("depthnet-console:   failed to initialize depthNet\n");
		return 0;
	}

	// parse the desired colormap
	const cudaColormapType colormap = cudaColormapFromStr(cmdLine.GetString("colormap"));

	// parse the desired filter mode
	const cudaFilterMode filterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));


	/*
	 * load image from disk
	 */
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("depthnet-console:  failed to load image '%s'\n", imgFilename);
		return 0;
	}


	/*
	 * allocate output depth map
	 */
	float* depthCPU  = NULL;
	float* depthCUDA = NULL;

	if( !cudaAllocMapped((void**)&depthCPU, (void**)&depthCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("depthnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}


	/*
	 * perform the depth mapping
	 */
	if( !net->Process(imgCUDA, depthCUDA, imgWidth, imgHeight, colormap, filterMode) )
	{
		printf("depthnet-console:  failed to process depth map\n");
		return 0;
	}

	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// print out performance info
	net->PrintProfilerTimes();


	/*
	 * save output image
	 */
	if( !saveImageRGBA(outFilename, (float4*)depthCPU, imgWidth, imgHeight) )
		printf("depthnet-console:  failed to save output image to '%s'\n", outFilename);
	else
		printf("depthnet-console:  completed saving '%s'\n", outFilename);

	
	/*
	 * destroy resources
	 */
	printf("depthnet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgCPU));
	CUDA(cudaFreeHost(depthCPU));

	SAFE_DELETE(net);

	printf("depthnet-console:  shutdown complete\n");
	return 0;
}
