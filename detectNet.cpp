/*
 * http://github.com/dusty-nv/jetson-inference
 */
 
#include "detectNet.h"
#include "cudaMappedMemory.h"
#include "cudaResize.h"


#define OUTPUT_CVG  0
#define OUTPUT_BBOX 1



// constructor
detectNet::detectNet() : tensorNet()
{
	mCoverageThreshold = 0.5f;
}


// destructor
detectNet::~detectNet()
{
	
}


// Create
detectNet* detectNet::Create( const char* prototxt, const char* model, const char* mean_binary, const char* input_blob, const char* coverage_blob, const char* bbox_blob )
{
	detectNet* net = new detectNet();
	
	if( !net )
		return NULL;
	
	std::vector<std::string> output_blobs;
	output_blobs.push_back(coverage_blob);
	output_blobs.push_back(bbox_blob);
	
	if( !net->LoadNetwork(prototxt, model, mean_binary, input_blob, output_blobs) )
	{
		printf("detectNet -- failed to initialize.\n");
		return NULL;
	}
	
	return net;
}


cudaError_t cudaPreImageNet( float4* input, size_t inputWidth, size_t inputHeight, float* output, size_t outputWidth, size_t outputHeight, const float3& mean_value );



struct float6 { float x; float y; float z; float w; float v; float u; };
static inline float6 make_float6( float x, float y, float z, float w, float v, float u ) { float6 f; f.x = x; f.y = y; f.z = z; f.w = w; f.v = v; f.u = u; return f; }


inline static bool rectOverlap(const float6& r1, const float6& r2)
{
    return ! ( r2.x > r1.z  
        || r2.z < r1.x
        || r2.y > r1.w
        || r2.w < r1.y
        );
}

static void mergeRect( std::vector<float6>& rects, const float6& rect )
{
	const uint32_t num_rects = rects.size();
	
	bool intersects = false;
	
	for( uint32_t r=0; r < num_rects; r++ )
	{
		if( rectOverlap(rects[r], rect) )
		{
			intersects = true;   printf("found overlap\n");
			
			//if( rects[r].x > rect.x ) 
			
			break;
		}
			
	} 
	
	if( !intersects )
		rects.push_back(rect);
}


// Detect
bool detectNet::Detect( float* rgba, uint32_t width, uint32_t height, float* boundingBoxes, int* numBoxes, float* confidence )
{
	if( !rgba || width == 0 || height == 0 || !boundingBoxes || !numBoxes || *numBoxes < 1 )
	{
		printf("detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return false;
	}

	
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
	{
		printf("detectNet::Classify() -- cudaPreImageNet failed\n");
		return false;
	}
	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[OUTPUT_CVG].CUDA, mOutputs[OUTPUT_BBOX].CUDA };
	
	mContext->execute(1, inferenceBuffers);

	// cluster detection bboxes
	float* net_cvg   = mOutputs[OUTPUT_CVG].CPU;
	float* net_rects = mOutputs[OUTPUT_BBOX].CPU;
	
	const int ow  = mOutputs[OUTPUT_BBOX].dims.w;		// number of columns in bbox grid in X dimension
	const int oh  = mOutputs[OUTPUT_BBOX].dims.h;		// number of rows in bbox grid in Y dimension
	const int owh = ow * oh;							// total number of bbox in grid
	const int cls = GetNumClasses();					// number of object classes in coverage map
	
	const float cell_width  = /*width*/ mInputDims.w / ow;
	const float cell_height = /*height*/ mInputDims.h / oh;
	
	const float scale_x = float(width) / float(mInputDims.w);
	const float scale_y = float(height) / float(mInputDims.h);
	
	printf("cell width %f  height %f\n", cell_width, cell_height);
	printf("scale x %f  y %f\n", scale_x, scale_y);
	
	std::vector< std::vector<float6> > rects;
	rects.resize(cls);
	
	for( uint32_t z=0; z < cls; z++ )
	{
		rects[z].reserve(owh);
		
		for( uint32_t y=0; y < oh; y++ )
		{
			for( uint32_t x=0; x < ow; x++)
			{
				const float coverage = net_cvg[z * owh + y * ow + x];
				
				if( coverage > mCoverageThreshold )
				{
					const float mx = x * cell_width;
					const float my = y * cell_height;
					
					const float x1 = (net_rects[0 * owh + y * ow + x] + mx) * scale_x;		// left
					const float y1 = (net_rects[1 * owh + y * ow + x] + my) * scale_y;		// top
					const float x2 = (net_rects[2 * owh + y * ow + x] + mx) * scale_x;		// right
					const float y2 = (net_rects[3 * owh + y * ow + x] + my) * scale_y;		// bottom
					
					printf("rect x=%u y=%u  cvg=%f  %f %f   %f %f\n", x, y, coverage, x1, x2, y1, y2);
					
					mergeRect( rects[z], make_float6(x1, y1, x2, y2, coverage, z) );
				}
			}
		}
	}
	
	printf("done clustering rects\n");
	
	const uint32_t numMax = *numBoxes;
	int n = 0;
	
	for( uint32_t z = 0; z < cls; z++ )
	{
		printf("z = %u\n", z);
		
		const uint32_t numBox = rects[z].size();
		
		for( uint32_t b = 0; b < numBox && n < numMax; b++ )
		{
			printf("b = %u\n", b );
			const float6 r = rects[z][b];
			
			boundingBoxes[n * 4 + 0] = r.x;
			boundingBoxes[n * 4 + 1] = r.y;
			boundingBoxes[n * 4 + 2] = r.z;
			boundingBoxes[n * 4 + 3] = r.w;
			
			if( confidence != NULL )
			{
				confidence[n * 2 + 0] = r.v;	// coverage
				confidence[n * 2 + 1] = r.u;	// class ID
			}
			
			n++;
		}
	}
	
	*numBoxes = n;
	return true;
}




// DrawBoxes
bool detectNet::DrawBoxes( float* input, float* output, uint32_t width, uint32_t height, const float* boundingBoxes, const int numBoxes, const float color[4] )
{
	if( !input || !output || width == 0 || height == 0 || !boundingBoxes || numBoxes < 1 )
		return false;
	
	return true;
}
	
	
