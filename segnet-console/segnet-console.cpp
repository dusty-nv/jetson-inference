/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "segNet.h"

#include "loadImage.h"
#include "cudaMappedMemory.h"


#include <sys/time.h>


uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}


// main entry point
int main( int argc, char** argv )
{
	printf("segnet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("segnet-console:   input image filename required\n");
		return 0;
	}

	if( argc < 3 )
	{
		printf("segnet-console:   output image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	const char* outFilename = argv[2];
	const char* modelName   = "fcn-alexnet-cityscapes";

	if( argc > 3 )
		modelName = argv[3];	

	segNet::NetworkType type = segNet::SEGNET_CUSTOM;

	if( strcasecmp(modelName, "fcn-alexnet-cityscapes") == 0 )
		type = segNet::FCN_ALEXNET_CITYSCAPES_21;
	else if( strcasecmp(modelName, "fcn-alexnet-pascal-voc") == 0 )
		type = segNet::FCN_ALEXNET_PASCAL_VOC;
	else if( strcasecmp(modelName, "fcn-alexnet-synthia-cvpr16") == 0 )
		type = segNet::FCN_ALEXNET_SYNTHIA_CVPR16;
	else if( strcasecmp(modelName, "fcn-alexnet-synthia-summer") == 0 )
		type = segNet::FCN_ALEXNET_SYNTHIA_SUMMER;


	// create segnet
	segNet* net = segNet::Create(type);

	if( !net )
	{
		printf("segnet-console:   failed to initialize segnet\n");
		return 0;
	}
	
	net->EnableProfiler();

	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 0;
	int    imgHeight = 0;
		
	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
	{
		printf("failed to load image '%s'\n", imgFilename);
		return 0;
	}

	// allocate output image
	float* outCPU  = NULL;
	float* outCUDA = NULL;

	if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )
	{
		printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
		return 0;
	}

	printf("segnet-console:  beginning processing overlay (%zu)\n", current_timestamp());

	// process image overlay
	net->SetGlobalAlpha(120);

	if( !net->Overlay(imgCUDA, outCUDA, imgWidth, imgHeight) )
	{
		printf("segnet-console:  failed to process segmentation overlay.\n");
		return 0;
	}

	printf("segnet-console:  finished processing overlay  (%zu)\n", current_timestamp());

	// save output image
	if( !saveImageRGBA(outFilename, (float4*)outCPU, imgWidth, imgHeight) )
		printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
	else
		printf("segnet-console:  completed saving '%s'\n", outFilename);

	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	CUDA(cudaFreeHost(outCPU));
	delete net;
	return 0;
}
