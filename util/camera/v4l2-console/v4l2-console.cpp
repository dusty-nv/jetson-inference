/*
 * inference-101
 */

#include "v4l2Camera.h"

#include <stdio.h>
#include <signal.h>
//#include <unistd.h>
#include <QImage>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}



int main( int argc, char** argv )
{
	printf("v4l2-console\n  args (%i):  ", argc);
	
	/*
	 * verify parameters
	 */
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
	if( argc < 2 )
	{
		printf("v4l2-console:  0 arguments were supplied.\n");
		printf("usage:  v4l2-console <filename>\n");
		printf("      ./v4l2-console /dev/video0\n");
		
		return 0;
	}
	
	const char* dev_path = argv[1];
	printf("v4l2-console:   attempting to initialize video device '%s'\n\n", dev_path);
	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	 */
	v4l2Camera* camera = v4l2Camera::Create(dev_path);
	
	if( !camera )
	{
		printf("\nv4l2-console:  failed to initialize video device '%s'\n", dev_path);
		return 0;
	}
	
	printf("\nv4l2-console:  successfully initialized video device '%s'\n", dev_path);
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n", camera->GetPixelDepth());
	
	
	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\nv4l2-console:  failed to open camera '%s' for streaming\n", dev_path);
		return 0;
	}
	
	printf("\nv4l2-console:  camera '%s' open for streaming\n", dev_path);
	
	
	while( !signal_recieved )
	{
		uint8_t* img = (uint8_t*)camera->Capture(500);
		
		if( !img )
		{
			//printf("got NULL image from camera capture\n");
			continue;
		}
		else
		{
			printf("recieved new video frame\n");
			
			static int num_frames = 0;
			
			const int width  = camera->GetWidth();
			const int height = camera->GetHeight();
			
			QImage qImg(width, height, QImage::Format_RGB32);
			
			for( int y=0; y < height; y++ )
			{
				for( int x=0; x < width; x++ )
				{
					const int value = img[y * width + x];
					if( value != 0 )
						printf("%i %i  %i\n", x, y, value);
					qImg.setPixel(x, y, qRgb(value, value, value));
				}
			}
			
			char output_filename[64];
			sprintf(output_filename, "camera-%u.jpg", num_frames);
			
			qImg.save(QString(output_filename));
			num_frames++;
		}
			
	}
	
	printf("\nv4l2-console:  un-initializing video device '%s'\n", dev_path);
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}
	
	printf("v4l2-console:  video device '%s' has been un-initialized.\n", dev_path);
	printf("v4l2-console:  this concludes the test of video device '%s'\n", dev_path);
	return 0;
}