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

#include "depthWindow.h"
#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

int usage()
{
	printf("usage: depth-viewer [-h] [--camera CAMERA] file_left file_right\n");
	printf("                    [--width WIDTH] [--height HEIGHT] [--calibration CAL]\n");
	printf("                    [--depth DEPTHNET] [--stereo STEREONET]\n");
	printf("                    [--colormap COLORMAP] [--filter-mode MODE]\n");
	printf("                    [--segmentation SEGMENTATION] [--alpha alpha]\n");
	printf("3D depth viewer using mono or stereo depth DNN, with segmentation\n\n");
	printf("positional arguments:\n");
	printf("  file_left         filename of the left input image to process\n");
	printf("  file_right        filename of the left input image to process\n");
	printf("optional arguments:\n");
	printf("  --help            show this help message and exit\n");
	printf("  --camera CAMERA   index of the MIPI CSI camera to use (e.g. CSI camera 0),\n");
	printf("                    or for VL42 cameras the /dev/video device to use.\n");
     printf("                    by default, MIPI CSI camera 0 will be used.\n");
	printf("  --width WIDTH     desired width of camera stream (default: 1280 pixels)\n");
	printf("  --height HEIGHT   desired height of camera stream (default: 720 pixels)\n");
	printf("  --calibration CAL intrinsic calibration file(s) to use for the cameras\n");
	printf("  --depth DEPTHNET     pre-trained mono depth model to use\n");
	printf("  --stereo STEREONET   pre-trained stereo depth model to use\n");
	printf("  --colormap COLORMAP  colormap to use (default is 'viridis')\n");
	printf("  --filter-mode MODE   filtering mode used during visualization,\n");
	printf("                       options are 'point' or 'linear' (default: 'linear')\n");
	printf("  --segmentation SEGM  pre-trained segmentation model to use\n");
	printf("  --alpha ALPHA        segmentation overlay alpha blending (default: 120)\n\n");

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
		printf("\ncan't catch SIGINT\n");


	/*
	 * create depth window
	 */
	DepthWindow* depthWindow = DepthWindow::Create(cmdLine);

	if( !depthWindow )
	{
		printf("depth-viewer:  failed to open DepthWindow\n");
		return 0;
	}


	/*
	 * create control window
	 */
	/*ControlWindow* controlWindow = ControlWindow::Create(cmdLine, depthWindow);

	if( !controlWindow )
	{
		printf("depth-viewer:  failed to open ControlWindow\n");
		return 0;
	}*/


	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture & render latest frame
		depthWindow->Render();

		// update the control window
		//controlWindow->ProcessEvents();

		// check if the user quit
		if( depthWindow->IsClosed() /*|| controlWindow->IsClosed()*/ )
			signal_recieved = true;
	}
	

	/*
	 * destroy resources
	 */
	printf("depth-viewer:  shutting down...\n");
	
	if( depthWindow != NULL )
		delete depthWindow;

	//if( captureWindow != NULL )
	//	delete captureWindow;

	printf("depth-viewer:  shutdown complete.\n");
	return 0;
}

