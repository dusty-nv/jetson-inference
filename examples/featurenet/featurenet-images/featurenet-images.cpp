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
#include "imageIO.h"


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


typedef uchar3 pixelType;        // this can be uchar3, uchar4, float3, float4


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
	const uint32_t maxFeatures = cmdLine.GetUnsignedInt("max-features", net->GetMaxFeatures());
	
	
	const int numFeatures = net->Match(images[0], width[0], height[0], imageFormatFromType<pixelType>(),
								images[1], width[1], height[1], imageFormatFromType<pixelType>(),
								NULL, NULL, NULL, threshold, true);
	
		
	CUDA(cudaDeviceSynchronize());
	
	net->PrintProfilerTimes();
	
#if 0
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image image
		uchar3* image = NULL;

		if( !input->Capture(&image, 1000) )
		{
			// check for EOS
			if( !input->IsStreaming() )
				break;

			LogError("featurenet-images:  failed to capture next frame\n");
			continue;
		}

		skipped += 1;
		
		if( skipped % frameskip == 0 )
		{
			img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
			skipped = 0;
			
			if( img_class >= 0 )
				LogVerbose("featurenet-images:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class, net->GetClassDesc(img_class));	
			else
				LogError("featurenet-images:  failed to classify frame\n");
		}

		if( img_class >= 0 )
		{
			char str[256];
			sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
			font->OverlayText(image, input->GetWidth(), input->GetHeight(),
						   str, 5, 5, make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));
		}	

		// render outputs
		if( output != NULL )
		{
			output->Render(image, input->GetWidth(), input->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
			output->SetStatus(str);	

			// check if the user quit
			if( !output->IsStreaming() )
				signal_recieved = true;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
#endif	
	
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

