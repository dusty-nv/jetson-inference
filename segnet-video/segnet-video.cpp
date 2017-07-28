/*
 * http://github.com/dusty-nv/jetson-inference
 */
#include "loadImage.h"
#include "segNet.h"

#include "glDisplay.h"
#include "glTexture.h"

#include "commandLine.h"
#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include <sys/time.h>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)

using namespace cv;
using namespace std;

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

bool saveImageRGBA_video(VideoWriter& outputVideo,float4** outCPU,Mat &mat,int imgWidth, int imgHeight)
{
	float* cpuPtr = (float*)*outCPU;
	for( uint32_t y=0; y < imgHeight; y++ )
	{
		for( uint32_t x=0; x < imgWidth; x++ )
		{
			mat.at<Vec3b>(y,x)[0]=cpuPtr[y*imgWidth+x+2];
			mat.at<Vec3b>(y,x)[1]=cpuPtr[y*imgWidth+x+1];
			mat.at<Vec3b>(y,x)[2]=cpuPtr[y*imgWidth+x];
		}
	}
	outputVideo.write(mat);
	return true;
}

// main entry point
int main( int argc, char** argv )
{
	printf("Usage:\n./segnet-video video_path path_to_save_video\n\n");
	printf("segnet-video\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);

	printf("\n\n");


	// retrieve filename arguments
	if( argc < 2 )
	{
		printf("segnet-video:   input video filename required\n");
		return 0;
	}

	if( argc < 3 )
	{
		printf("segnet-video:   output video filename required\n");
		return 0;
	}

	const char* imgFilename = argv[1];
	const char* outFilename = argv[2];


	// create the segNet from pretrained or custom model by parsing the command line
	segNet* net = segNet::Create(argc, argv);

	if( !net )
	{
		printf("segnet-console:   failed to initialize segnet\n");
		return 0;
	}

	// enable layer timings for the console application
	net->EnableProfiler();

	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;

	//=====================================================
	// load video
	VideoCapture cap(imgFilename);
	if(!cap.isOpened())  // check if we succeeded
	{
		printf("failed to load video '%s'\n", imgFilename);
		return 0;
	}

	imgWidth=cap.get(CV_CAP_PROP_FRAME_WIDTH);
	imgHeight=cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	const size_t   imgSize   = imgWidth * imgHeight * sizeof(float) * 4;
	if( !cudaAllocMapped((void**)&imgCPU, (void**)&imgCUDA, imgSize) )
	{
		printf(LOG_CUDA "failed to allocated %zu bytes for image %s\n", imgSize, imgFilename);
		return false;
	}

	//write video
	int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC));     // Get Codec Type- Int form
	// Transform from int to char via Bitwise operators
	char EXT[] = {(char)(ex & 0XFF) , (char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24), 0};
	Size S = Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
			(int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	VideoWriter outputVideo;
	outputVideo.open(outFilename, ex, 20/*caps[0].get(CV_CAP_PROP_FPS)*/, S, true);
	if (!outputVideo.isOpened())
	{
		cout  << "Could not open the output video for write: "<<outFilename  << endl;
	}
	cout << "Input codec type: " << EXT << endl;

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

	// allocate output image
	float* outCPU  = NULL;
	float* outCUDA = NULL;

	if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}

	//	printf("segnet-console:  beginning processing overlay (%zu)\n", current_timestamp());

	// set alpha blending value for classes that don't explicitly already have an alpha	
	net->SetGlobalAlpha(120);

	//!loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight)
	while(loadImageRGBA_video(cap,(float4**)&imgCPU,mat,imgWidth,imgHeight))
	{
		// process image overlay
		if( !net->Overlay(imgCUDA, outCUDA, imgWidth, imgHeight) )
		{
			printf("segnet-console:  failed to process segmentation overlay.\n");
			return 0;
		}

		// save output image
		if( !saveImageRGBA_video(outputVideo, (float4**)outCPU, imgWidth, imgHeight) )
			printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
		else
			printf("segnet-console:  completed saving '%s'\n", outFilename);

		if( display != NULL )
		{
			char str[256];
			sprintf(str, "TensorRT build %x | %s | %04.1f FPS", NV_GIE_VERSION, net->HasFP16() ? "FP16" : "FP32", display->GetFPS());
			//sprintf(str, "GIE build %x | %s | %04.1f FPS | %05.2f%% %s", NV_GIE_VERSION, net->GetNetworkName(), display->GetFPS(), confidence * 100.0f, net->GetClassDesc(img_class));
			display->SetTitle(str);
		}

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)outCUDA, make_float2(0.0f, 255.0f),
						(float4*)outCUDA, make_float2(0.0f, 1.0f),
						imgWidth, imgHeight));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, outCUDA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);
				//printf("%u\t",((char*)imgRGBA)[0]);
			}

			display->EndRender();
		}

	}


	//printf("segnet-console:  finished processing overlay  (%zu)\n", current_timestamp());



	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	CUDA(cudaFreeHost(outCPU));
	delete net;
	return 0;
}
