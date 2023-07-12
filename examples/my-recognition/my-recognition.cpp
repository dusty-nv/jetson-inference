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

// include imageNet header for image recognition
#include <jetson-inference/imageNet.h>

// include loadImage header for loading images
#include <jetson-utils/loadImage.h>


// main entry point
int main( int argc, char** argv )
{
	// a command line argument containing the image filename is expected,
	// so make sure we have at least 2 args (the first arg is the program)
	if( argc < 2 )
	{
		printf("my-recognition:  expected image filename as argument\n");
		printf("example usage:   ./my-recognition my_image.jpg\n");
		return 0;
	}

	// retrieve the image filename from the array of command line args
	const char* imgFilename = argv[1];

	// these variables will store the image data pointer and dimensions
	uchar3* imgPtr = NULL;   // shared CPU/GPU pointer to image
	int imgWidth   = 0;      // width of the image (in pixels)
	int imgHeight  = 0;      // height of the image (in pixels)
		
	// load the image from disk as uchar3 RGB (24 bits per pixel)
	if( !loadImage(imgFilename, &imgPtr, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	// load the GoogleNet image recognition network with TensorRT
	// you can use "resnet18", "resnet50", "alexnet", ect. instead
	imageNet* net = imageNet::Create("googlenet");

	// check to make sure that the network model loaded properly
	if( !net )
	{
		printf("failed to load image recognition network\n");
		return 0;
	}

	// this variable will store the confidence of the classification (between 0 and 1)
	float confidence = 0.0;

	// classify the image, return the object class index (or -1 on error)
	const int classIndex = net->Classify(imgPtr, imgWidth, imgHeight, &confidence);

	// make sure a valid classification result was returned	
	if( classIndex >= 0 )
	{
		// retrieve the name/description of the object class index
		const char* classDescription = net->GetClassDesc(classIndex);

		// print out the classification results
		printf("image is recognized as '%s' (class #%i) with %f%% confidence\n", 
			  classDescription, classIndex, confidence * 100.0f);
	}
	else
	{
		// if Classify() returned < 0, an error occurred
		printf("failed to classify image\n");
	}
	
	// free the network's resources before shutting down
	delete net;

	// this is the end of the example!
	return 0;
}

