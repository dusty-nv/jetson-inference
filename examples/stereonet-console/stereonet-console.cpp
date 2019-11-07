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

#include "stereoNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"


int usage()
{
	printf("usage: stereonet-console [-h] [--network NETWORK]\n");
	printf("                         [--colormap COLORMAP] [--filter-mode MODE]\n");
	printf("                         file_left file_right file_out\n\n");
	printf("Stereo disparity depth estimation from left/right images using stereoNet DNN.\n\n");
	printf("positional arguments:\n");
	printf("  file_left            filename of the left input image to process\n");
	printf("  file_right           filename of the left input image to process\n");
	printf("  file_out             filename of the output depth image to save\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
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
	 * parse filename arguments
	 */
	const char* imgFilename[] = { cmdLine.GetPosition(0), cmdLine.GetPosition(1) };
	const char* depthFilename = cmdLine.GetPosition(2);

	if( !imgFilename[0] || !imgFilename[1] )
	{
		printf("stereonet-console:   left and right input filenames required\n");
		return usage();
	}

	tensorNet::EnableVerbose();

	/*
	 * create stereo depth network
	 */
	stereoNet* net = stereoNet::Create(stereoNet::RESNET18_2D); //(argc, argv);

	if( !net )
	{
		printf("stereonet-console:   failed to initialize stereoNet\n");
		return 0;
	}

	// parse the desired colormap
	const cudaColormapType colormap = cudaColormapFromStr(cmdLine.GetString("colormap"));

	// parse the desired filter mode
	const cudaFilterMode filterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));


	/*
	 * load images from disk
	 */
	float* imgInput[] = { NULL, NULL };
	int    imgWidth   = 0;
	int    imgHeight  = 0;
		
	for( int n=0; n < 2; n++ )
	{
		if( !loadImageRGBA(imgFilename[n], (float4**)&imgInput[n], &imgWidth, &imgHeight) )
		{
			printf("stereonet-console:  failed to load image '%s'\n", imgFilename[n]);
			return 0;
		}
	}


	/*
	 * allocate output depth map
	 */
	float* imgDepth = NULL;

	if( !cudaAllocMapped((void**)&imgDepth, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("stereonet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}


	/*
	 * perform the depth mapping
	 */
	if( !net->Process(imgInput[0], imgInput[1], imgDepth, imgWidth, imgHeight, colormap, filterMode) )
	{
		printf("stereonet-console:  failed to process depth map\n");
		return 0;
	}

	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// print out performance info
	net->PrintProfilerTimes();


	/*
	 * save output depth image
	 */
	if( depthFilename != NULL )
	{
		if( !saveImageRGBA(depthFilename, (float4*)imgDepth, imgWidth, imgHeight) )
			printf("stereonet-console:  failed to save output image to '%s'\n", depthFilename);
		else
			printf("stereonet-console:  completed saving '%s'\n", depthFilename);
	}

#if 0
	/*
	 * save point cloud
	 */
	const char* pointCloudFilename = cmdLine.GetString("point-cloud");

	if( pointCloudFilename != NULL )
	{
		printf("stereonet-console:  saving point cloud to '%s'\n", pointCloudFilename);
		
		if( !net->SavePointCloud(pointCloudFilename, imgCPU, imgWidth, imgHeight, cmdLine.GetString("calibration")) )
			printf("stereonet-console:  failed to save point cloud to '%s'\n", pointCloudFilename);
	}
#endif

	/*
	 * destroy resources
	 */
	printf("stereonet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgInput[0]));
	CUDA(cudaFreeHost(imgInput[1]));
	CUDA(cudaFreeHost(imgDepth));

	SAFE_DELETE(net);

	printf("stereonet-console:  shutdown complete\n");
	return 0;
}
