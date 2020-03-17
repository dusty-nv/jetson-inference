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

#include "odometryNet.h"

#include "cudaMappedMemory.h"
#include "cudaWarp.h"

#include "loadImage.h"
#include "commandLine.h"
#include "mat33.h"


// print usage
int print_usage()
{
	printf("usage: odometry-console [-h] [--network NETWORK]\n");
	printf("                        file_A file_B\n\n");
	printf("Perform visual odometry estimation on a sequential pair of images\n\n");
	printf("positional arguments:\n");
	printf("  file_A               filename of the first input image to process\n");
	printf("  file_B               filename of the second input image to process\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
	printf("%s\n", odometryNet::Usage());

	return 0;
}


// main entry point
int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetPositionArgs() < 2 )
	{
		printf("odometry-console:   two input image filenames required\n");
		return print_usage();
	}

	const char* imgPath[] = { cmdLine.GetPosition(0), cmdLine.GetPosition(1) };
	

	/*
	 * load network
	 */
	odometryNet* net = odometryNet::Create(argc, argv);

	if( !net )
	{
		printf("odometry-console:  failed to load network\n");
		return 0;
	}


	/* 
	 * load input images
	 */
	float4* imgInput[]  = { NULL, NULL };
	int     imgWidth[]  = { 0, 0 };
	int     imgHeight[] = { 0, 0 };

	for( uint32_t n=0; n < 2; n++ )
	{
		if( !loadImageRGBA(imgPath[n], (float4**)&imgInput[n], &imgWidth[n], &imgHeight[n]) )
		{
			printf("odometry-console:  failed to load image #%u '%s'\n", n, imgPath[n]);
			return 0;
		}
	}

	// verify images have the same size
	if( imgWidth[0] != imgWidth[1] || imgHeight[0] != imgHeight[1] )
	{
		printf("odometry-console:  the two images must have the same dimensions\n");
		return 0;
	}


	/*
	 * estimate the odometry with the network
	 */
	if( !net->Process(imgInput[0], imgInput[1], imgWidth[0], imgHeight[0]) )
	{
		printf("odometry-console:  failed to find homography\n");
		return 0;
	}

	// print out odometry info
	const uint32_t numOutputs = net->GetNumOutputs();
	const float* outputs = net->GetOutput();

	printf("odometry:  ");

	for( uint32_t n=0; n < numOutputs; n++ )
		printf("%f ", outputs[n]);

	printf("\n\n");

	// print out performance info
	net->PrintProfilerTimes();


	/*
	 * destroy resources
	 */
	printf("odometry-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgInput[0]));
	CUDA(cudaFreeHost(imgInput[1]));

	SAFE_DELETE(net);

	printf("odometry-console:  shutdown complete\n");
	return 0;
}

