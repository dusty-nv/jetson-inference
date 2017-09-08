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

#include "loadImage.h"
#include "cudaFont.h"



// main entry point
int main( int argc, char** argv )
{
	printf("imagenet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("imagenet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create imageNet
	imageNet* net = imageNet::Create(argc, argv);

	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
		return 0;
	}
	
	net->EnableProfiler();
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	float confidence = 0.0f;
	
	// classify image
	const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
	
	if( img_class >= 0 )
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, net->GetClassDesc(img_class));
	
		if( argc > 2 )
		{
			const char* outputFilename = argv[2];
			
			// overlay the classification on the image
			cudaFont* font = cudaFont::Create();
			
			if( font != NULL )
			{
				char str[512];
				sprintf(str, "%2.3f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));

				const int overlay_x = 10;
				const int overlay_y = 10;
				const int px_offset = overlay_y * imgWidth * 4 + overlay_x * 4;

				// if the image has a white background, use black text (otherwise, white)
				const float white_cutoff = 225.0f;
				bool white_background = false;

				if( imgCPU[px_offset] > white_cutoff && imgCPU[px_offset + 1] > white_cutoff && imgCPU[px_offset + 2] > white_cutoff )
					white_background = true;

				// overlay the text on the image
				font->RenderOverlay((float4*)imgCUDA, (float4*)imgCUDA, imgWidth, imgHeight, (const char*)str, 10, 10,
								white_background ? make_float4(0.0f, 0.0f, 0.0f, 255.0f) : make_float4(255.0f, 255.0f, 255.0f, 255.0f));
			}
			
			printf("imagenet-console:  attempting to save output image to '%s'\n", outputFilename);
			
			if( !saveImageRGBA(outputFilename, (float4*)imgCPU, imgWidth, imgHeight) )
				printf("imagenet-console:  failed to save output image to '%s'\n", outputFilename);
			else
				printf("imagenet-console:  completed saving '%s'\n", outputFilename);
		}
	}
	else
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
