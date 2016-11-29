/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "segNet.h"

#include "cudaMappedMemory.h"
#include "cudaOverlay.h"
#include "cudaResize.h"



// Create
segNet* segNet::Create( NetworkType networkType )
{
	if( networkType == FCN_ALEXNET_PASCAL_VOC )
		return Create("FCN-Alexnet-PASCAL-VOC/deploy.prototxt", "FCN-Alexnet-PASCAL-VOC/snapshot_iter_146400.caffemodel", "FCN-Alexnet-PASCAL-VOC/pascal-voc-classes.txt", NULL );
	else if( networkType == FCN_ALEXNET_SYNTHIA_CVPR16 )
		return Create("FCN-Alexnet-SYNTHIA-CVPR16/deploy.prototxt", "FCN-Alexnet-SYNTHIA-CVPR16/snapshot_iter_1206700.caffemodel", "FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-labels.txt", "FCN-Alexnet-SYNTHIA-CVPR16/synthia-cvpr16-train-colors.txt" );
	else if( networkType == FCN_ALEXNET_SYNTHIA_SUMMER )
		return Create("FCN-Alexnet-SYNTHIA-Summer/deploy.prototxt", "FCN-Alexnet-SYNTHIA-Summer/snapshot_iter_529956.caffemodel", "FCN-Alexnet-SYNTHIA-Summer/synthia-seq-labels.txt", "FCN-Alexnet-SYNTHIA-Summer/synthia-seq-train-colors.txt" );	
	else if( networkType == FCN_ALEXNET_CITYSCAPES_21 )
		return Create("FCN-Alexnet-Cityscapes-21/deploy.prototxt", "FCN-Alexnet-Cityscapes-21/snapshot_iter_114865.caffemodel", "FCN-Alexnet-Cityscapes-21/cityscapes-labels.txt", "FCN-Alexnet-Cityscapes-21/cityscapes-training-colors.txt" );	
	else
		return NULL;
}
	

// constructor
segNet::segNet() : tensorNet()
{
	mClassColors[0] = NULL;	// cpu ptr
	mClassColors[1] = NULL;  // gpu ptr
}


// destructor
segNet::~segNet()
{
	
}


// Create
segNet* segNet::Create( const char* prototxt, const char* model, const char* labels_path, const char* colors_path, const char* input_blob, const char* output_blob )
{
	// create segmentation model
	segNet* net = new segNet();
	
	if( !net )
		return NULL;
	
	//net->EnableProfiler();	
	net->EnableDebug();
	//net->DisableFP16();		// debug;

	// load network
	std::vector<std::string> output_blobs;
	output_blobs.push_back(output_blob);
	
	if( !net->LoadNetwork(prototxt, model, NULL, input_blob, output_blobs) )
	{
		printf("segNet -- failed to initialize.\n");
		return NULL;
	}
	
	// initialize array of class colors
	const uint32_t numClasses = net->GetNumClasses();
	
	if( !cudaAllocMapped((void**)&net->mClassColors[0], (void**)&net->mClassColors[1], numClasses * sizeof(float4)) )
		return NULL;
	
	for( uint32_t n=0; n < numClasses; n++ )
	{
		net->mClassColors[0][n*4+0] = 255.0f;	// r
		net->mClassColors[0][n*4+1] = 0.0f;	// g
		net->mClassColors[0][n*4+2] = 0.0f;	// b
		net->mClassColors[0][n*4+3] = 255.0f;	// a
	}
	
	// load class info
	net->loadClassColors(colors_path);
	net->loadClassLabels(labels_path);
	
	return net;
}


// loadClassColors
bool segNet::loadClassColors( const char* filename )
{
	if( !filename )
		return false;
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", filename);
		return false;
	}
	
	char str[512];
	int  idx = 0;

	while( fgets(str, 512, f) != NULL )
	{
		const int len = strlen(str);
		
		if( len > 0 )
		{
			if( str[len-1] == '\n' )
				str[len-1] = 0;

			int r = 255;
			int g = 255;
			int b = 255;

			sscanf(str, "%i %i %i", &r, &g, &b);
			printf("segNet -- class %02i  color %i %i %i\n", idx, r, g, b);
			SetClassColor(idx, r, g, b);
			idx++; 
		}
	}
	
	fclose(f);
	
	printf("segNet -- loaded %i class colors\n", idx);
	
	if( idx == 0 )
		return false;
	
	return true;
}


// loadClassLabels
bool segNet::loadClassLabels( const char* filename )
{
	if( !filename )
		return false;
	
	FILE* f = fopen(filename, "r");
	
	if( !f )
	{
		printf("segNet -- failed to open %s\n", filename);
		return false;
	}
	
	char str[512];

	while( fgets(str, 512, f) != NULL )
	{
		const int len = strlen(str);
		
		if( len > 0 )
		{
			if( str[len-1] == '\n' )
				str[len-1] = 0;

			printf("segNet -- class %02zu  label '%s'\n", mClassLabels.size(), str);
			mClassLabels.push_back(str);
		}
	}
	
	fclose(f);
	
	printf("segNet -- loaded %zu class labels\n", mClassLabels.size());
	
	if( mClassLabels.size() == 0 )
		return false;
	
	return true;
}


// SetClassColor
void segNet::SetClassColor( uint32_t classIndex, float r, float g, float b, float a )
{
	if( classIndex >= GetNumClasses() || !mClassColors[0] )
		return;
	
	const uint32_t i = classIndex * 4;
	
	mClassColors[0][i+0] = r;
	mClassColors[0][i+1] = g;
	mClassColors[0][i+2] = b;
	mClassColors[0][i+3] = a;
}



// FindClassID
int segNet::FindClassID( const char* label_name )
{
	if( !label_name )
		return -1;

	const uint32_t numLabels = mClassLabels.size();

	for( uint32_t n=0; n < numLabels; n++ )
	{
		if( strcasecmp(label_name, mClassLabels[n].c_str()) == 0 )
			return n;
	}

	return -1;
}



// declaration from imageNet.cu
cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight );



// Overlay
bool segNet::Overlay( float* rgba, float* output, uint32_t width, uint32_t height, float alpha, const char* ignore_class )
{
	if( !rgba || width == 0 || height == 0 || !output )
	{
		printf("segNet::Overlay( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	printf("segnet network width %u height %u\n", mWidth, mHeight);
	

	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight)) )
	{
		printf("segNet::Overlay() -- cudaPreImageNet failed\n");
		return false;
	}

	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{
		printf(LOG_GIE "segNet::Overlay() -- failed to execute tensorRT context\n");
		return false;
	}


	// retrieve scores
	float* scores = mOutputs[0].CPU;

	const int s_w = mOutputs[0].dims.w;
	const int s_h = mOutputs[0].dims.h;
	const int s_c = mOutputs[0].dims.c;
		
	const float s_x = float(width) / float(s_w);
	const float s_y = float(height) / float(s_h);
	

	// if desired, find the ID of the class to ignore (typically void)
	const int ignoreID = FindClassID(ignore_class);
	
	printf(LOG_GIE "segNet::Overlay -- s_w %i  s_h %i  s_c %i  s_x %f  s_y %f\n", s_w, s_h, s_c, s_x, s_y);
	printf(LOG_GIE "segNet::Overlay -- ignoring class '%s' id=%i\n", ignore_class, ignoreID);


	// overlay pixels onto original
	for( uint32_t y=0; y < s_h; y++ )
	{
		for( uint32_t x=0; x < s_w; x++ )
		{
			float p_max[3] = {-100000.0f, -100000.0f, -100000.0f };
			int   c_max[3] = { -1, -1, -1 };

			for( uint32_t c=0; c < s_c; c++ )	// classes
			{
				const float p = scores[c * s_w * s_h + y * s_w + x];

				if( c_max[0] < 0 || p > p_max[0] )
				{
					p_max[0] = p;
					c_max[0] = c;
				}
				else if( c_max[1] < 0 || p > p_max[1] )
				{
					p_max[1] = p;
					c_max[1] = c;
				}
				else if( c_max[2] < 0 || p > p_max[2] )
				{
					p_max[2] = p;
					c_max[2] = c;
				}
			}

			/*printf("%02u %u  class %i  %f  %s  class %i  %f  %s  class %i  %f  %s\n", x, y, 
				   c_max[0], p_max[0], (c_max[0] >= 0 && c_max[0] < GetNumClasses()) ? GetClassLabel(c_max[0]) : " ", 
				   c_max[1], p_max[1], (c_max[1] >= 0 && c_max[1] < GetNumClasses()) ? GetClassLabel(c_max[1]) : " ",
				   c_max[2], p_max[2], (c_max[2] >= 0 && c_max[2] < GetNumClasses()) ? GetClassLabel(c_max[2]) : " ");
			*/
	   
			const int oy = y * s_y;
			const int ox = x * s_x;
			
			float* c_color = GetClassColor(c_max[0] == ignoreID ? c_max[1] : c_max[0]);
			
			for( int yy=oy; yy < oy + s_y; yy++ )
			{
				for( int xx=ox; xx < ox + s_x; xx++ )
				{
					float* px_in  = rgba +   (((yy * width * 4) + xx * 4));
					float* px_out = output + (((yy * width * 4) + xx * 4));
					
					const float alph = /*c_color[3]*/ alpha / 255.0f;
					const float inva = 1.0f - alph;

					px_out[0] = alph * c_color[0] + inva * px_in[0];
					px_out[1] = alph * c_color[1] + inva * px_in[1];
					px_out[2] = alph * c_color[2] + inva * px_in[2];
					px_out[3] = 255.0f;
				}
			}
		}
	}

	return true;
}


	
	
