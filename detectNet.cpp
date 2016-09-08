/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "detectNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"



// constructor
detectNet::detectNet()
{

}


// destructor
detectNet::~detectNet()
{
	
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, const char* mean_binary, const char* input_blob, const char* output_blob )
{
	detectNet* net = new detectNet();
	
	if( !net )
		return NULL;
	
	if( !net->LoadNetwork(prototxt, model, mean_binary, input_blob, output_blob) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}

					
// Detect
int detectNet::Detect( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	return 0;
}

