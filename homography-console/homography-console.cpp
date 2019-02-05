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

#include "homographyNet.h"

#include "loadImage.h"
#include "commandLine.h"

#include <sys/time.h>
#include <iostream>


// retrieve current timestamp
uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}


// print usage
int print_usage()
{
	printf("\nUSAGE:\n");
	printf("  homography-console --imageA=<path> --imageB=<path> --model=<name/path>\n\n");
	printf("  note:  --model is optional and can be path to ONNX model, 'coco', or 'webcam'\n");
	printf("         if --model is left unspecified, the default model is 'webcam'\n");
 
     return 0;
}

// main entry point
int main( int argc, char** argv )
{
	// parse command line
	commandLine cmdLine(argc, argv);

	const char* imgPath[] = { cmdLine.GetString("imageA"),
						 cmdLine.GetString("imageB") };

	if( !imgPath[0] || !imgPath[1] )
	{
		printf("homography-console:   two input image filenames required\n");
		return print_usage();
	}
	
	
	// load network
	homographyNet* net = homographyNet::Create(argc, argv);

	if( !net )
	{
		printf("homography-console:  failed to load network\n");
		return 0;
	}

	net->EnableProfiler();


	// load images
	float* imgCPU[] = { NULL, NULL };
	float* imgCUDA[] = { NULL, NULL };
	int    imgWidth[] = { 0, 0 };
	int    imgHeight[] = { 0, 0 };

	for( uint32_t n=0; n < 2; n++ )
	{
		if( !loadImageRGBA(imgPath[n], (float4**)&imgCPU[n], (float4**)&imgCUDA[n], &imgWidth[n], &imgHeight[n]) )
		{
			printf("homography-console:  failed to load image #%u '%s'\n", n, imgPath[n]);
			return 0;
		}
	}

	// verify images have the same size
	if( imgWidth[0] != imgWidth[1] || imgHeight[0] != imgHeight[1] )
	{
		printf("homography-console:  the two images must have the same dimensions\n");
		return 0;
	}


	// process the network
	for( int n=0; n < 10; n++ )
	{
		if( !net->FindHomography(imgCUDA[0], imgCUDA[1], imgWidth[0], imgHeight[0]) )
		{
			printf("homography-console:  failed to find homography\n");
			return 0;
		}
	}

	delete net;
	return 0;
}

