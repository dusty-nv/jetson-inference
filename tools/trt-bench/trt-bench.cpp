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


uint64_t imagesProcessed[NUM_DEVICES];


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
void* process( void* param )
{
	if( !param )
	{
		printf("NULL thread parameter, exiting thread\n");
		return NULL;
	}

	imageNet* net = (imageNet*)param;
	deviceType dev = net->GetDevice();
	const char* str = deviceTypeToStr(dev);

	printf("%s thread started\n", str);

	while( !signal_recieved )
	{
		if( !net->Process() )
			printf("%s network failed to process\n", str);

		imagesProcessed[dev]++;

		//printf("images %s %lu\n", str, imagesProcessed[dev]);
	}

	printf("exiting %s thread\n", str);
}


// main entry point
int main( int argc, char** argv )
{
	printf("\ntrt-bench usage: --image=<path> [--GPU=FP16|INT8] [--DLA_0=FP16] [--DLA_1=FP16] [--allowGPUFallback]\n");


	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * set CUDA driver to spin-wait
	 */
	CUDA(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
	cudaFree(0);


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
	printf("   -- allowGPUFallback:  %s\n\n", allowGPUFallback ? "ON" : "OFF");


	/*
	 * load image from disk
	 */
	float* imgInput  = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgPath, (float4**)&imgInput, &imgWidth, &imgHeight) )
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
#if 0
		imageNet* net = imageNet::Create( "networks/alexnet_noprob.prototxt", "networks/bvlc_alexnet.caffemodel", 
								    NULL, "networks/ilsvrc12_synset_words.txt", 
								    "data", "fc8", 1 /*MAX_BATCH_SIZE_DEFAULT*/,
								    precisions[n], (deviceType)n, allowGPUFallback );
#else
		imageNet* net = imageNet::Create( "networks/googlenet_noprob.prototxt", "networks/bvlc_googlenet.caffemodel", 
							    		NULL, "networks/ilsvrc12_synset_words.txt", 
							    		"data", "loss3/classifier", 1 /*MAX_BATCH_SIZE_DEFAULT*/,
							    		precisions[n], (deviceType)n, allowGPUFallback );
#endif

		if( !net )
		{
			printf("failed to create network for device %s\n", deviceTypeToStr((deviceType)n));
			continue;
		}

		// pre-process the image into NCHW format
		if( !net->PreProcess(imgInput, imgWidth, imgHeight) )
			printf("imageNet::PreProcess() failed for device %s\n", deviceTypeToStr((deviceType)n)); 

		// put the networks on their own streams for concurrent execution
		net->CreateStream();

		networks.push_back(net);
	}

	const int numNetworks = networks.size();
	memset(imagesProcessed, 0, sizeof(imagesProcessed));


	/*
	 * spin up threads
	 */
	std::vector<Thread*> threads;
	
	for( int n=0; n < numNetworks; n++ )
	{
		Thread* thread = new Thread();
		imageNet* net  = networks[n];

		if( !thread )
		{
			printf("failed to allocate %s thread\n", deviceTypeToStr(net->GetDevice()));
			continue;
		}

		threads.push_back(thread);

		if( !thread->StartThread(&process, (void*)net) )
			printf("failed to start %s thread\n", deviceTypeToStr(net->GetDevice()));
	}
		

	/*
	 * run until user quits
	 */
	timespec timeBegin = timestamp();

	while( !signal_recieved )
	{
		sleep(1);
		
		uint64_t totalImages = 0;

		for( int n=0; n < NUM_DEVICES; n++ )
			totalImages += imagesProcessed[n];

		const timespec timeNow = timestamp();
		const timespec timeElapsed = timeDiff(timeBegin, timeNow);

		const double seconds = timeElapsed.tv_sec + double(timeElapsed.tv_nsec)*double(1e-9);
		const double imagesPerSec = double(totalImages) / seconds;

		printf("%f img/sec  (", imagesPerSec);

		for( int n=0; n < numNetworks; n++ )
		{
			const deviceType dev = networks[n]->GetDevice();
			printf("%s %f img/sec", deviceTypeToStr(dev), double(imagesProcessed[dev]) / seconds);

			if( n < numNetworks - 1 )
				printf(", ");
		}

		printf(")\n");
		
		if( timeElapsed.tv_sec >= 5 )
		{
			timeBegin = timestamp();
			memset(imagesProcessed, 0, sizeof(imagesProcessed));
		}
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
	CUDA(cudaFreeHost(imgInput));

	for( int n=0; n < networks.size(); n++ )
		delete networks[n];

	return 0;
}
