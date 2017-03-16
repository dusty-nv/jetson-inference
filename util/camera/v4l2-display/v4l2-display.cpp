/*
 * inference-101
 */

#include "v4l2Camera.h"
#include "glDisplay.h"
#include "cudaMappedMemory.h"

#include <stdio.h>


int main( int argc, char** argv )
{
	printf("v4l2-display\n  args (%i):  ", argc);
	
	/*
	 * verify parameters
	 */
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
	if( argc < 2 )
	{
		printf("v4l2-display:  0 arguments were supplied.\n");
		printf("usage:  v4l2-display <filename>\n");
		printf("      ./v4l2-display /dev/video0\n");
		
		return 0;
	}
	
	const char* dev_path = argv[1];
	printf("v4l2-display:   attempting to initialize video device '%s'\n\n", dev_path);
	
	
	/*
	 * create the camera device
	 */
	v4l2Camera* camera = v4l2Camera::Create(dev_path);
	
	if( !camera )
	{
		printf("\nv4l2-display:  failed to initialize video device '%s'\n", dev_path);
		return 0;
	}
	
	printf("\nv4l2-display:  successfully initialized video device '%s'\n", dev_path);
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n", camera->GetPixelDepth());
	
	printf("\nv4l2-display:  un-initializing video device '%s'\n", dev_path);
	
	
	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
	{
		printf("\nv4l2-display:  failed to create openGL display\n");
		return 0;
	}
	
	glTexture* tex = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_LUMINANCE8);
	
	if( !tex )
	{
		printf("v4l2-display:  failed to create %ux%u openGL texture\n", camera->GetWidth(), camera->GetHeight());
		return 0;
	}
	
	printf("v4l2-display:  initialized %u x %u openGL texture (%u bytes)\n", tex->GetWidth(), tex->GetHeight(), tex->GetSize());
	
	
	

	/*
	 * shutdown
	 */
	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}
	
	printf("v4l2-display:  video device '%s' has been un-initialized.\n", dev_path);
	printf("v4l2-display:  this concludes the test of video device '%s'\n", dev_path);
	return 0;
}