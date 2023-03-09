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

#include "videoSource.h"
#include "videoOutput.h"

#include "cudaFont.h"
#include "actionNet.h"

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
	printf("usage: actionnet [--help] [--network=NETWORK] ...\n");
	printf("                input_URI [output_URI]\n\n");
	printf("Classify the action/activity of an image sequence.\n");
	printf("See below for additional arguments that may not be shown above.\n\n");	
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --network=NETWORK pre-trained model to load (see below for options)\n");
	printf("positional arguments:\n");
	printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
	printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");

	printf("%s", actionNet::Usage());
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
		LogError("actionnet:  failed to create input stream\n");
		return 1;
	}


	/*
	 * create output stream
	 */
	videoOutput* output = videoOutput::Create(cmdLine, ARG_POSITION(1));
	
	if( !output )
	{
		LogError("actionnet:  failed to create output stream\n");	
		return 1;
	}
	

	/*
	 * create font for image overlay
	 */
	cudaFont* font = cudaFont::Create();
	
	if( !font )
	{
		LogError("actionnet:  failed to load font for overlay\n");
		return 1;
	}


	/*
	 * create recognition network
	 */
	actionNet* net = actionNet::Create(cmdLine);
	
	if( !net )
	{
		LogError("actionnet:  failed to initialize actionNet\n");
		return 1;
	}


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

		// classify the action sequence
		float confidence = 0.0f;
		const int class_id = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);

		if( class_id >= 0 )
			LogVerbose("actionnet:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, class_id, net->GetClassDesc(class_id));	
		else
			LogError("actionnet:  failed to classify frame\n");

		// overlay the results
		if( class_id >= 0 )
		{
			char str[256];
			sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(class_id));
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
				break;
		}

		// print out timing info
		net->PrintProfilerTimes();
	}
	
	
	/*
	 * destroy resources
	 */
	LogVerbose("actionnet:  shutting down...\n");
	
	SAFE_DELETE(input);
	SAFE_DELETE(output);
	SAFE_DELETE(net);
	
	LogVerbose("actionnet:  shutdown complete.\n");
	return 0;
}

