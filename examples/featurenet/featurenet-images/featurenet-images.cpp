/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "featureNet.h"
#include "cudaWarp.h"
#include "imageIO.h"
#include "mat33.h"

#include <algorithm>


int usage()
{
	printf("usage: featurenet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Classify a video/image stream using an image recognition DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", featureNet::Usage());
	printf("%s", Log::Usage());

	return 0;
}


typedef uchar4 pixelType;        // this can be uchar3, uchar4, float3, float4


int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();

	const uint32_t numPositionArgs = cmdLine.GetPositionArgs();
	
	if( numPositionArgs < 2 )
	{
		LogError("featurenet-images:  must specify at least two input image filenames\n\n");
		return usage();
	}
	
	
	/*
	 * load input images
	 */
	pixelType* images[] = {NULL, NULL};
	
	int width[] = {0,0};
	int height[] = {0,0};
	
	for( uint32_t n=0; n < 2; n++ )
	{
		if( !loadImage(cmdLine.GetPosition(n), &images[n], &width[n], &height[n]) )
			return 0;
	}


	/*
	 * load feature matching network
	 */
	featureNet* net = featureNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("featurenet-images:  failed to initialize featureNet\n");
		return 0;
	}

	const float threshold = cmdLine.GetFloat("threshold", FEATURENET_DEFAULT_THRESHOLD);
	const float drawScale = cmdLine.GetFloat("draw-scale", FEATURENET_DEFAULT_DRAWING_SCALE);
	const int maxFeatures = cmdLine.GetUnsignedInt("max-features", net->GetMaxFeatures());
	
	
	/*
      * match features
	 */
	float2 features[2][1200];
	float  confidence[1200];
	
	const int numFeatures = net->Match(images[0], width[0], height[0], imageFormatFromType<pixelType>(),
								images[1], width[1], height[1], imageFormatFromType<pixelType>(),
								features[0], features[1], confidence, threshold, true);
	
	if( numFeatures < 0 )
	{
		LogError("featurenet-images:  failed to process feature extraction/matching\n");
		return 0;
	}
	
	for( int n=0; n < numFeatures; n++ )
	{
		printf("match %i   %f  (%f, %f) -> (%f, %f)\n", n, confidence[n], features[0][n].x, features[0][n].y, features[1][n].x, features[1][n].y);
	}
	
	// draw features
	for( int n=0; n < 2; n++ )
	{
		//printf("drawing image %i\n", n);
		
		net->DrawFeatures(images[n], width[n], height[n], imageFormatFromType<pixelType>(),
					   features[n], std::min(numFeatures, maxFeatures), false,
					   drawScale, make_float4(0,255,0,255));
					   
		if( numPositionArgs > n+2 )
			saveImage(cmdLine.GetPosition(n+2), images[n], width[n], height[n]);
	}


	/*
	 * find homography
	 */
	float H[3][3];
	float H_inv[3][3];
	
	if( !net->FindHomography(features[0], features[1], numFeatures, H, H_inv) )
	{
		LogError("featurenet-images:  failed to find homography\n");
		return 0;
	}
	
	mat33_print(H, "H");	
	mat33_print(H_inv, "H_inv");
	
	
	/*
	 * warp images
	 */
	CUDA(cudaWarpPerspective(images[1], width[1], height[1],
						images[0], width[0], height[0],
						H, true));
	
	if( numPositionArgs > 4 )
		saveImage(cmdLine.GetPosition(4), images[0], width[0], height[0]);
	
	CUDA(cudaDeviceSynchronize());
	net->PrintProfilerTimes();
	

	/*
	 * destroy resources
	 */
	LogVerbose("featurenet-images:  shutting down...\n");

	SAFE_DELETE(net);
	
	for( uint32_t n=0; n < 2; n++ )
		CUDA_FREE_HOST(images[n]);
	
	LogVerbose("featurenet-images:  shutdown complete.\n");
	
	return 0;
}

