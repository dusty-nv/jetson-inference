/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"



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
	detectNet* net = detectNet::Create("/home/ubuntu/ped-100/deploy.prototxt", "/home/ubuntu/ped-100/snapshot_iter_70800.caffemodel", "/home/ubuntu/ped-100/mean.binaryproto" );
	
	if( !net )
	{
		printf("detectnet-console:   failed to initialize detectNet\n");
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

	float confidence = 0.0f;
	
	// classify image
	const int img_class = net->Detect(imgCUDA, imgWidth, imgHeight, &confidence);
	
	if( img_class < 0 )
		printf("detectnet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	else
		printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");
	
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
