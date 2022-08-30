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
			return 1;
	}


	/*
	 * load feature matching network
	 */
	featureNet* net = featureNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("featurenet-images:  failed to initialize featureNet\n");
		return 1;
	}

	const float threshold = cmdLine.GetFloat("threshold", FEATURENET_DEFAULT_THRESHOLD);
	const float drawScale = cmdLine.GetFloat("draw-scale", FEATURENET_DEFAULT_DRAWING_SCALE);
	const int maxFeatures = cmdLine.GetUnsignedInt("max-features", net->GetMaxFeatures());
	
	
	/*
      * match features
	 */
	float2* features[] = {NULL, NULL};
	float* confidence = NULL;
	
	const int numFeatures = net->Match(images[0], width[0], height[0],
								images[1], width[1], height[1],
								&features[0], &features[1], 
								&confidence, threshold, true);
	
	if( numFeatures < 0 )
	{
		LogError("featurenet-images:  failed to process feature extraction/matching\n");
		return 1;
	}
	
	for( int n=0; n < numFeatures && n < maxFeatures; n++ )
	{
		printf("match %i   %f  (%f, %f) -> (%f, %f)\n", n, confidence[n], features[0][n].x, features[0][n].y, features[1][n].x, features[1][n].y);
	}
	
	// draw features
	for( int n=0; n < 2; n++ )
	{
		net->DrawFeatures(images[n], width[n], height[n], features[n], 
					   std::min(numFeatures, maxFeatures), true,
					   drawScale, make_float4(0,255,0,255));
					   
		if( numPositionArgs > n+2 )
			saveImage(cmdLine.GetPosition(n+2), images[n], width[n], height[n]);
	}

#if 0
	// DEBUG
	features[0][0].x = 208.0; features[0][0].y = 144.0;
	features[1][0].x = 320.0; features[1][0].y = 144.0;
	features[0][1].x = 256.0; features[0][1].y = 160.0;
	features[1][1].x = 368.0; features[1][1].y = 160.0;
	features[0][2].x = 256.0; features[0][2].y = 192.0;
	features[1][2].x = 384.0; features[1][2].y = 192.0;
	features[0][3].x = 256.0; features[0][3].y = 304.0;
	features[1][3].x = 400.0; features[1][3].y = 320.0;
	features[0][4].x = 176.0; features[0][4].y = 160.0;
	features[1][4].x = 288.0; features[1][4].y = 160.0;
	features[0][5].x = 272.0; features[0][5].y = 304.0;
	features[1][5].x = 416.0; features[1][5].y = 320.0;
	features[0][6].x = 208.0; features[0][6].y = 160.0;
	features[1][6].x = 320.0; features[1][6].y = 160.0;
	features[0][7].x = 224.0; features[0][7].y = 144.0;
	features[1][7].x = 336.0; features[1][7].y = 144.0;
	features[0][8].x = 240.0; features[0][8].y = 208.0;
	features[1][8].x = 368.0; features[1][8].y = 208.0;
	features[0][9].x = 352.0; features[0][9].y = 416.0;
	features[1][9].x = 368.0; features[1][9].y = 432.0;
	features[0][10].x = 224.0; features[0][10].y = 208.0;
	features[1][10].x = 352.0; features[1][10].y = 208.0;
	features[0][11].x = 256.0; features[0][11].y = 208.0;
	features[1][11].x = 384.0; features[1][11].y = 208.0;
	features[0][12].x = 240.0; features[0][12].y = 176.0;
	features[1][12].x = 352.0; features[1][12].y = 176.0;
	features[0][13].x = 240.0; features[0][13].y = 192.0;
	features[1][13].x = 368.0; features[1][13].y = 192.0;
	features[0][14].x = 240.0; features[0][14].y = 304.0;
	features[1][14].x = 384.0; features[1][14].y = 320.0;
	features[0][15].x = 160.0; features[0][15].y = 192.0;
	features[1][15].x = 272.0; features[1][15].y = 192.0;
	features[0][16].x = 272.0; features[0][16].y = 192.0;
	features[1][16].x = 400.0; features[1][16].y = 192.0;
	features[0][17].x = 112.0; features[0][17].y = 304.0;
	features[1][17].x = 256.0; features[1][17].y = 320.0;
	features[0][18].x = 192.0; features[0][18].y = 144.0;
	features[1][18].x = 304.0; features[1][18].y = 144.0;
	features[0][19].x = 160.0; features[0][19].y = 160.0;
	features[1][19].x = 272.0; features[1][19].y = 160.0;
#endif

	/*
	 * find homography
	 */
	float H[3][3];
	float H_inv[3][3];
	
	if( !net->FindHomography(features[0], features[1], std::min(numFeatures, maxFeatures), H, H_inv) )
	{
		LogError("featurenet-images:  failed to find homography\n");
		return 1;
	}
	

	/*mat33_identity(H);
	mat33_shear(H, H, 0.5f, 0.0f);
	mat33_scale(H, H, 0.5f, 0.5f);*/
	/*mat33_identity(H);
	mat33_rotation(H, 90.0f, width[1] * 0.5f, height[1] * 0.5f);
	mat33_translate(H, H, -200.0f, 000.0f);
	mat33_inverse(H_inv, H);*/
	
	mat33_print(H, "H");	
	mat33_print(H_inv, "H_inv");
	
	float2 transformed_coords[] = {
		make_float2(0.0f, 0.0f),
		make_float2(width[1], 0.0f),
		make_float2(width[1], height[1]),
		make_float2(0.0f, height[1])
	};
	
	printf("original image corners:\n");
	
	for( int n=0; n < 4; n++ )
		printf("  (%f, %f)\n", transformed_coords[n].x, transformed_coords[n].y);
	
	printf("transformed image corners:\n");
	
	for( int n=0; n < 4; n++ )
	{
		mat33_transform(transformed_coords[n].x, transformed_coords[n].y, 
					 transformed_coords[n].x, transformed_coords[n].y, H_inv);
					 
		printf("  (%f, %f)\n", transformed_coords[n].x, transformed_coords[n].y);
	}
	
	//mat33_transpose(H, H);
	//mat33_transpose(H_inv, H_inv);
	
	/*
	 * warp images
	 * TODO image[1] should be same size as image[0] ???
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

