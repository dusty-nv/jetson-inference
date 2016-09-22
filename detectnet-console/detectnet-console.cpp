/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"


// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("detectnet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create detectNet
	//detectNet* net = detectNet::Create("ped-100/deploy.prototxt", "ped-100/snapshot_iter_70800.caffemodel", "ped-100/mean.binaryproto" );
	detectNet* net = detectNet::Create("multiped-90/deploy.prototxt", "multiped-90/snapshot_iter_32040.caffemodel", "multiped-90/mean.binaryproto" );
	
	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
		return 0;
	}
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU  = NULL;
	float* bbCUDA = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-console:  failed to alloc output memory\n");
		return 0;
	}
	
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
	
	// classify image
	int numBoundingBoxes = maxBoxes;
	
	if( !net->DetectRGBA(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU) )
		printf("detectnet-console:  failed to classify '%s'\n", imgFilename);
	//else
		//printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");
	
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
