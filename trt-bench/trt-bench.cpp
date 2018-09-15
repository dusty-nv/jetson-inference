/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "commandLine.h"
#include "timespec.h"
#include "Thread.h"

#include <signal.h>
#include <unistd.h>


uint64_t imagesProcessed = 0;


// exit handler
bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


// thread entry
static void* process( void* param )
{
	if( !param )
	{
		printf("NULL thread parameter, exiting thread\n");
		return NULL;
	}

	imageNet* net = (imageNet*)param;
	printf("%s thread started\n", deviceTypeToStr(net->GetDevice()));

	while( !signal_recieved )
	{
		//printf("processing %s\n", deviceTypeToStr(net->GetDevice()));

		if( !net->Process() )
			printf("%s network failed to process\n", deviceTypeToStr(net->GetDevice()));

		imagesProcessed++;
	}

	printf("exiting %s thread\n", deviceTypeToStr(net->GetDevice()));
}


// main entry point
int main( int argc, char** argv )
{
	printf("trt-bench\n");


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	// get the input image filename
	const char* imgPath = cmdLine.GetString("image");

	if( !imgPath )
	{
		printf("path to input image must be specified as --image=<path>\n");
		return 0;
	}

	// determine requested devices
	precisionType precisions[NUM_DEVICES];

	for( int n=0; n < NUM_DEVICES; n++ )
	{
		const char* deviceName = deviceTypeToStr((deviceType)n);
		
		precisions[n] = precisionTypeFromStr(cmdLine.GetString(deviceName));

		if( precisions[n] == TYPE_DISABLED && cmdLine.GetFlag(deviceName) )
			precisions[n] = TYPE_FASTEST;

		printf("   -- %s: %s\n", deviceName, precisionTypeToStr(precisions[n]));
	}

	// determine if GPU fallback is requested
	const bool allowGPUFallback = cmdLine.GetFlag("allowGPUFallback");
	printf("   -- allowGPUFallback:  %s\n", allowGPUFallback ? "ON" : "OFF");


	/*
	 * load image from disk
	 */
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgPath, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgPath);
		return 0;
	}


	/*
	 * load networks
	 */
	std::vector<imageNet*> networks;

	for( int n=0; n < NUM_DEVICES; n++ )
	{
		if( precisions[n] == TYPE_DISABLED )
			continue;

		// create network instance on the specified device
		imageNet* net = imageNet::Create( imageNet::ALEXNET, /*MAX_BATCH_SIZE_DEFAULT*/ 1,
								    precisions[n], (deviceType)n, allowGPUFallback );

		if( !net )
		{
			printf("failed to create network for device %s\n", deviceTypeToStr((deviceType)n));
			continue;
		}

		// pre-process the image into NCHW format
		if( !net->PreProcess(imgCUDA, imgWidth, imgHeight) )
			printf("imageNet::PreProcess() failed for device %s\n", deviceTypeToStr((deviceType)n)); 

		networks.push_back(net);
	}

	const int numNetworks = networks.size();


	/*
	 * spin up threads
	 */
	std::vector<Thread*> threads;
	
	for( int n=0; n < numNetworks; n++ )
	{
		Thread* thread = new Thread();

		if( !thread )
		{
			printf("failed to allocate %s thread\n", deviceTypeToStr(networks[n]->GetDevice()));
			continue;
		}

		threads.push_back(thread);

		if( !thread->StartThread(&process, (void*)networks[n]) )
			printf("failed to start %s thread\n", deviceTypeToStr(networks[n]->GetDevice()));
	}
		

	/*
	 * run until user quits
	 */
	const timespec timeBegin = timestamp();

	while( !signal_recieved )
	{
		sleep(1);
		
		const timespec timeNow = timestamp();
		const timespec timeElapsed = timeDiff(timeBegin, timeNow);

		const double seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec)*double(1e-9);
		const double imagesPerSec = double(imagesProcessed) / seconds;

		printf("%f images per second  (%lu images in %f seconds)\n", imagesPerSec, imagesProcessed, seconds);
	}


	/*
	 * wait for threads to stop
	 */
	printf("\nwaiting for threads to stop...\n");
	sleep(1);
	printf("shutting down...\n");


	/*
	 * free resources
	 */
	CUDA(cudaFreeHost(imgCPU));

	for( int n=0; n < networks.size(); n++ )
		delete networks[n];

	return 0;
}
