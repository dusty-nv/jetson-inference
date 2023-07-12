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

#include "videoSource.h"
#include "videoOutput.h"

#include "cudaFont.h"
#include "imageNet.h"

#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogVerbose("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: imagenet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Classify a video/image stream using an image recognition DNN.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network=NETWORK pre-trained model to load (see below for options)\n");
	printf("  --topK=N   	   show the topK number of class predictions (default: 1)\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", imageNet::Usage());
	printf("%s", videoSource::Usage());
	printf("%s", videoOutput::Usage());
	printf("%s", Log::Usage());

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
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * create input stream
	 */
	videoSource* input = videoSource::Create(cmdLine, ARG_POSITION(0));

	if( !input )
	{
		LogError("imagenet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("imagenet:  failed to create output stream\n");	
		return 1;
	}
	

	/*
	 * create font for image overlay
	 */
	cudaFont* font = cudaFont::Create();
	
	if( !font )
	{
		LogError("imagenet:  failed to load font for overlay\n");
		return 1;
	}


	/*
	 * create recognition network
	 */
	imageNet* net = imageNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("imagenet:  failed to initialize imageNet\n");
		return 1;
	}

	const int topK = cmdLine.GetInt("topK", 1);  // by default, get only the top result
	
	
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture next image
		uchar3* image = NULL;
		int status = 0;
		
		if( !input->Capture(&image, &status) )
		{
			if( status == videoSource::TIMEOUT )
				continue;
			
			break; // EOS
		}

		// classify image - note that if you only want the top class, you can simply run this instead:
		// 	float confidence = 0.0f;
		//	const int img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
		imageNet::Classifications classifications;	// std::vector<std::pair<uint32_t, float>>  (classID, confidence)

		if( net->Classify(image, input->GetWidth(), input->GetHeight(), classifications, topK) < 0 )
			continue;
		
		// draw predicted class labels
		for( uint32_t n=0; n < classifications.size(); n++ )
		{
			const uint32_t classID = classifications[n].first;
			const char* classLabel = net->GetClassLabel(classID);
			const float confidence = classifications[n].second * 100.0f;
			
			LogVerbose("imagenet:  %2.5f%% class #%i (%s)\n", confidence, classID, classLabel);	

			char str[256];
			sprintf(str, "%05.2f%% %s", confidence, classLabel);
			font->OverlayText(image, input->GetWidth(), input->GetHeight(),
						   str, 5, 5 + n * (font->GetSize() + 5), 
						   make_float4(255,255,255,255), make_float4(0,0,0,100));
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
				break;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("imagenet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	LogVerbose("imagenet:  shutdown complete.\n");
	return 0;
}

