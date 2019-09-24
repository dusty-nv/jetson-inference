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

#include "segNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"


int usage()
{
	printf("usage: segnet-console [-h] [--network NETWORK]\n");
	printf("                      [--visualize VISUAL] [--alpha ALPHA]\n");
	printf("                      [--filter-mode MODE] [--ignore-class CLASS]\n");
	printf("                      file_in file_out\n\n");
	printf("Segment and classify an image using a semantic segmentation DNN.\n\n");
	printf("positional arguments:\n");
	printf("  file_in              filename of the input image to process\n");
	printf("  file_out             filename of the output image to save\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
	printf("  --visualize VISUAL   visualization mode, options are: 'mask' or 'overlay'\n");
	printf("  --alpha ALPHA        overlay alpha blending, range 0-255 (default: 120)\n");
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
	 * parse filename arguments
	 */
	const char* imgFilename = cmdLine.GetPosition(0);
	const char* outFilename = cmdLine.GetPosition(1);

	if( !imgFilename || !outFilename )
	{
		printf("segnet-console:   input and output image filenames required\n");
		return usage();
	}


	/*
	 * create segmentation network
	 */
	segNet* net = segNet::Create(argc, argv);

	if( !net )
	{
		printf("segnet-console:   failed to initialize segnet\n");
		return 0;
	}

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetOverlayAlpha(cmdLine.GetFloat("alpha", 120.0f));

	// get the desired alpha blend filtering mode
	const segNet::FilterMode filterMode = segNet::FilterModeFromStr(cmdLine.GetString("filter-mode", "linear"));

	// get the object class to ignore (if any)
	const char* ignoreClass = cmdLine.GetString("ignore-class", "void");

	// get the visualization mode (mask or overlay)
	const char* visualization = cmdLine.GetString("visualize", "overlay");


	/*
	 * load image from disk
	 */
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}


	/*
	 * allocate output image
	 */
	float* outCPU  = NULL;
	float* outCUDA = NULL;

	if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}


	/*
	 * perform the segmentation
	 */
	if( !net->Process(imgCUDA, imgWidth, imgHeight, ignoreClass) )
	{
		printf("segnet-console:  failed to process segmentation\n");
		return 0;
	}

	// generate image overlay
	if( strcasecmp(visualization, "mask") == 0 )
	{
		if( !net->Mask(outCUDA, imgWidth, imgHeight, filterMode) )
		{
			printf("segnet-console:  failed to generate overlay.\n");
			return 0;
		}
	}
	else
	{
		if( !net->Overlay(outCUDA, imgWidth, imgHeight, filterMode) )
		{
			printf("segnet-console:  failed to generate overlay.\n");
			return 0;
		}
	}

	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// print out performance info
	net->PrintProfilerTimes();


	/*
	 * save output image
	 */
	if( !saveImageRGBA(outFilename, (float4*)outCPU, imgWidth, imgHeight) )
		printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
	else
		printf("segnet-console:  completed saving '%s'\n", outFilename);

	
	/*
	 * destroy resources
	 */
	printf("segnet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgCPU));
	CUDA(cudaFreeHost(outCPU));

	SAFE_DELETE(net);

	printf("segnet-console:  shutdown complete\n");
	return 0;
}
