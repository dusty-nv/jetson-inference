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

#include "flowNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"


int usage()
{
	printf("usage: flownet-console [-h] [--network NETWORK]\n");
	printf("                       file_A file_B file_out\n\n");
	printf("Dense optical flow estimation from image pairs, using flowNet DNN.\n\n");
	printf("positional arguments:\n");
	printf("  file_A               filename of the first input image\n");
	printf("  file_B               filename of the second input image\n");
	printf("  file_out             filename of the output image to save\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
	printf("  --filter-mode MODE   filtering mode used during visualization,\n");
	printf("                       options are 'point' or 'linear' (default: 'linear')\n\n");
	printf("%s\n", flowNet::Usage());

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
	const char* imgFilenames[] = { cmdLine.GetPosition(0),
							 cmdLine.GetPosition(1) };

	const char* outputFilename = cmdLine.GetPosition(2);

	if( !imgFilenames[0] || !imgFilenames[1] || !outputFilename )
	{
		printf("flownet-console:   two input images and one output image required\n");
		return usage();
	}


	/*
	 * create optical flow network
	 */
	flowNet* net = flowNet::Create(argc, argv);

	if( !net )
	{
		printf("flownet-console:   failed to initialize flowNet\n");
		return 0;
	}

	// parse the desired filter mode
	const cudaFilterMode filterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));


	/*
	 * load images from disk
	 */
	float* imgCPU[]  = { NULL, NULL };
	float* imgCUDA[] = { NULL, NULL };
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	for( int n=0; n < 2; n++ )
	{
		int width = 0;
		int height = 0;

		// load the next image in the sequence
		if( !loadImageRGBA(imgFilenames[n], (float4**)&imgCPU[n], (float4**)&imgCUDA[n], &width, &height) )
		{
			printf("flownet-console:  failed to load image '%s'\n", imgFilenames[n]);
			return 0;
		}

		// make sure the images have the same dimensions
		if( n == 0 )
		{
			imgWidth = width;
			imgHeight = height;
		}
		else if( width != imgWidth || height != imgHeight )
		{
			printf("flownet-console:  both images must have the same dimensions\n");
			return 0;
		}
	}


	/*
	 * allocate output flow map
	 */
	float* flowCPU  = NULL;
	float* flowCUDA = NULL;

	if( !cudaAllocMapped((void**)&flowCPU, (void**)&flowCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("flownet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}


	/*
	 * perform the optical flow
	 */
	if( !net->Process(imgCUDA[0], imgCUDA[1], flowCUDA, imgWidth, imgHeight, filterMode) )
	{
		printf("flownet-console:  failed to process optical flow\n");
		return 0;
	}

	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// print out performance info
	net->PrintProfilerTimes();


	/*
	 * save output image
	 */
	if( !saveImageRGBA(outputFilename, (float4*)flowCPU, imgWidth, imgHeight) )
		printf("flownet-console:  failed to save output image to '%s'\n", outputFilename);
	else
		printf("flownet-console:  completed saving '%s'\n", outputFilename);

	
	/*
	 * destroy resources
	 */
	printf("flownet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgCPU[0]));
	CUDA(cudaFreeHost(imgCPU[1]));
	CUDA(cudaFreeHost(flowCPU));

	SAFE_DELETE(net);

	printf("flownet-console:  shutdown complete\n");
	return 0;
}
