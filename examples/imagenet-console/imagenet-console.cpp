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

#include "imageNet.h"

#include "commandLine.h"
#include "loadImage.h"
#include "cudaFont.h"


int usage()
{
	printf("usage: imagenet-console [h] [--network NETWORK] file_in [file_out]\n\n");
	printf("Classify an image using an image recognition DNN.\n\n");
	printf("positional arguments:\n");
	printf("  file_in              filename of the input image to process\n");
	printf("  file_out             filename of the output image to save (optional)\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n\n");
	printf("%s\n", imageNet::Usage());

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
	 * check input filename
	 */
	const char* imgFilename = cmdLine.GetPosition(0);

	if( !imgFilename )
	{
		printf("imagenet-console:   input image filename required\n\n");
		return usage();
	}
	
	
	/*
	 * create recognition network
	 */
	imageNet* net = imageNet::Create(argc, argv);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}


	/*
	 * load image from disk
	 */
	uchar3* imgInput  = NULL;
	int     imgWidth  = 0;
	int     imgHeight = 0;
		
	if( !loadImage(imgFilename, &imgInput, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}
	

	/*
	 * classify image
	 */
	float confidence = 0.0f;
	const int img_class = net->Classify(imgInput, imgWidth, imgHeight, &confidence);
	
	// overlay the classification on the image
	if( img_class >= 0 )
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, net->GetClassDesc(img_class));
	
		const char* outputFilename = cmdLine.GetPosition(1);
		
		if( outputFilename != NULL )
		{
			// use font to draw the class description
			cudaFont* font = cudaFont::Create(adaptFontSize(imgWidth));
			
			if( font != NULL )
			{
				char str[512];
				sprintf(str, "%2.3f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));

				font->OverlayText(imgInput, imgWidth, imgHeight, (const char*)str, 10, 10,
							   make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));
			}

			// wait for GPU to complete work			
			CUDA(cudaDeviceSynchronize());

			// print out performance info
			net->PrintProfilerTimes();

			// save the output image to disk
			printf("imagenet-console:  attempting to save output image to '%s'\n", outputFilename);
			
			if( !saveImage(outputFilename, imgInput, imgWidth, imgHeight) )
				printf("imagenet-console:  failed to save output image to '%s'\n", outputFilename);
		}
	}
	else
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	

	/*
	 * destroy resources
	 */
	printf("imagenet-console:  shutting down...\n");

	CUDA(cudaFreeHost(imgInput));
	SAFE_DELETE(net);

	printf("imagenet-console:  shutdown complete\n");
	return 0;
}

