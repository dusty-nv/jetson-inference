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

#include "superResNet.h"

#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"



// print usage
int print_usage()
{
	printf("\nUSAGE:\n");
	printf("  superres-console --input=<path> --output=<path>\n\n");
	printf("     >  --input is a file path to the input image\n");
	printf("     >  --output is the path that the upscaled image will be written to\n");

     return 0;
}


// main entry point
int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	const char* inputPath = cmdLine.GetString("input");
	const char* outputPath = cmdLine.GetString("output");

	if( !inputPath || !outputPath )
	{
		printf("superres-console:  input and output image filenames required\n");
		return print_usage();
	}


	/*
	 * load super resolution network
	 */
	superResNet* net = superResNet::Create();

	if( !net )
	{
		printf("superres-console:  failed to load superResNet\n");
		return 0;
	}

	net->EnableLayerProfiler();


	/* 
	 * load input image
	 */
	float* inputCPU = NULL;
	float* inputCUDA = NULL;

	int inputWidth = 0;
	int inputHeight = 0;

	if( !loadImageRGBA(inputPath, (float4**)&inputCPU, (float4**)&inputCUDA, &inputWidth, &inputHeight) )
	{
		printf("superres-console:  failed to load input image '%s'\n", inputPath);
		return 0;
	}


	/*
	 * allocate memory for output
	 */
	float* outputCPU = NULL;
	float* outputCUDA = NULL;

	const int outputWidth = inputWidth * net->GetScaleFactor();
	const int outputHeight = inputHeight * net->GetScaleFactor();

	if( !cudaAllocMapped((void**)&outputCPU, (void**)&outputCUDA, outputWidth * outputHeight * sizeof(float4)) )
	{
		printf("superres-console:  failed to allocate memory for %ix%i output image\n", outputWidth, outputHeight);
		return 0;
	}

	printf("superres-console:  input image size - %ix%i\n", inputWidth, inputHeight);
	printf("superres-console:  output image size - %ix%i\n", outputWidth, outputHeight);


	/*
	 * upscale image with network
	 */
	for( int i=0; i < 10; i++ )
	{
		if( !net->UpscaleRGBA(inputCUDA, inputWidth, inputHeight,
						  outputCUDA, outputWidth, outputHeight) )
		{
			printf("superres-console:  failed to process super resolution network\n");
			return 0;
		}
	}	

	CUDA(cudaDeviceSynchronize());

	/*
	 * save output image
	 */
	printf("superres-console:  saving %ix%i output image to '%s'\n", outputWidth, outputHeight, outputPath);

	if( !saveImageRGBA(outputPath, (float4*)outputCPU, outputWidth, outputHeight) )
	{
		printf("superres-console:  failed to save output image to '%s'\n", outputPath);
		return 0;
	}

	delete net;
	return 0;
}


