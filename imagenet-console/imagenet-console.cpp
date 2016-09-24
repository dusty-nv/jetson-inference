/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"



// main entry point
int main( int argc, char** argv )
{
	printf("imagenet-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename argument
	if( argc < 2 )
	{
		printf("imagenet-console:   input image filename required\n");
		return 0;
	}
	
	const char* imgFilename = argv[1];
	

	// create imageNet
	imageNet* net = imageNet::Create();
	
	if( !net )
	{
		printf("imagenet-console:   failed to initialize imageNet\n");
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
	const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);
	
	if( img_class < 0 )
		printf("imagenet-console:  failed to classify '%s'  (result=%i)\n", imgFilename, img_class);
	else
	{
		printf("imagenet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, net->GetClassDesc(img_class));
	
		if( argc > 2 )
		{
			const char* outputFilename = argv[2];
			
			cudaFont* font = cudaFont::Create();
			
			if( font != NULL )
			{
				char str[512];
				sprintf(str, "%2.5f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
				font->RenderOverlay((float4*)imgCUDA, (float4*)imgCUDA, imgWidth, imgHeight, (const char*)str, 10, 10);
			}
			
			printf("imagenet-console:  attempting to save output image to '%s'\n", outputFilename);
			
			if( !saveImageRGBA(outputFilename, (float4*)imgCPU, imgWidth, imgHeight) )
				printf("imagenet-console:  failed to save output image to '%s'\n", outputFilename);
			else
				printf("imagenet-console:  completed saving '%s'\n", outputFilename);
		}
	}
	
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
