/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include <sys/time.h>
#include "opencv2/opencv.hpp"

#include <iostream>
#include <fstream>

using namespace cv;

bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}

uint64_t current_timestamp() {
	struct timeval te;
	gettimeofday(&te, NULL); // get current time
	return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

bool loadImageRGBA_video(VideoCapture& cap, float4** imgCPU, Mat &mat, int imgWidth, int imgHeight)
{
	float4* cpuPtr = *imgCPU;
	if (!cap.read(mat)) return false;
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			// opencv: bgr
			const float4 px = make_float4(float(mat.at<Vec3b>(y,x)[2]),
					float(mat.at<Vec3b>(y,x)[1]),
					float(mat.at<Vec3b>(y,x)[0]),
					0.0);

			cpuPtr[y*imgWidth+x] = px;
		}
	}
	return true;
}
// main entry point
int main( int argc, char** argv )
{
	printf("Usage:\n./detectnet-video video_path [path_to_save_boxes]\n\n");
	printf("detectnet-video\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");

	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	// retrieve filename argument
	if( argc < 2 )
	{
		printf("detectnet-video:   input video filename required\n");
		return 0;
	}

	const char* vidFilename = argv[1];


	// create detectNet
	detectNet* net = detectNet::Create(argc, argv);

	if( !net )
	{
		printf("detectnet-video:   failed to initialize detectNet\n");
		return 0;
	}

	//net->EnableProfiler();

	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();

	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;

	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
			!cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-video:  failed to alloc output memory\n");
		return 0;
	}

	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;

	//=====================================================
	// load video
	VideoCapture cap(vidFilename);
	if(!cap.isOpened())  // check if we succeeded
	{
		printf("failed to load video '%s'\n", vidFilename);
		return 0;
	}

	imgWidth=cap.get(CV_CAP_PROP_FRAME_WIDTH);
	imgHeight=cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 4;
	if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, vidFilename);
		return false;
	}


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;

	if( !display ) {
		printf("\ndetectnet-video:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(imgWidth, imgHeight, GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("detectnet-video:  failed to create openGL texture\n");
	}

	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();


	Mat mat;
	//loadImageRGBA_video(cap,(float4**)&imgCPU,img,imgWidth,imgHeight);

	/*
	//====================================================================
	if( !loadImageRGBA_video(cap,(float4**)&imgCPU,mat,imgWidth,imgHeight) )
	{
		printf("failed to load image '%s'\n", vidFilename);
		return 0;
	}
	 */
	int index=1;

	std::ofstream myfile;
	if(argc>2)
	{

		const char* csvFilename = argv[2];
		myfile.open (csvFilename);

		myfile<<"index, true,  left,   top,  right,   bottom\n";
	}
	while(loadImageRGBA_video(cap,(float4**)&imgCPU,mat,imgWidth,imgHeight))
	{


		// classify image
		int numBoundingBoxes = maxBoxes;

		printf("detectnet-video:  beginning processing network (%zu)\n", current_timestamp());

		const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);

		printf("detectnet-video:  finished processing network  (%zu)\n", current_timestamp());

		if( !result )
			printf("detectnet-video:  failed to classify '%s'\n", vidFilename);
		else 	// if the user supplied an output filename
		{
			printf("%i bounding boxes detected\n", numBoundingBoxes);

			int lastClass = 0;
			int lastStart = 0;

			if(numBoundingBoxes==0)
			{
				myfile<<index<<",0,0,0,0,0\n";
			}
			for( int n=0; n < numBoundingBoxes; n++ )
			{
				const int nc = confCPU[n*2+1];
				float* bb = bbCPU + (n * 4);

				printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]);
				if(argc>2)
				{
					myfile<<index<<","<<n+1<<","<<bb[0]<<","<< bb[1]<<","<< bb[2]<<","<< bb[3]<<"\n";
				}
				if( nc != lastClass || n == (numBoundingBoxes - 1) )
				{
					if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
						printf("detectnet-video:  failed to draw boxes\n");

					lastClass = nc;
					lastStart = n;
				}
			}

			CUDA(cudaThreadSynchronize());

			if( display != NULL )
			{
				char str[256];
				sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
				//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
				display->SetTitle(str);
			}
			/*
			// save image to disk
			printf("detectnet-console:  writing %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);

			if( !saveImageRGBA(argv[2], (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
				printf("detectnet-console:  failed saving %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
			else
				printf("detectnet-console:  successfully wrote %ix%i image to '%s'\n", imgWidth, imgHeight, argv[2]);
			 */
		}

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgCUDA, make_float2(0.0f, 255.0f),
						(float4*)imgCUDA, make_float2(0.0f, 1.0f),
						imgWidth, imgHeight));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgCUDA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);
				//printf("%u\t",((char*)imgRGBA)[0]);
			}

			display->EndRender();
		}
		index++;
	}
	if(argc>2)
	{
		myfile.close();
	}

	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
