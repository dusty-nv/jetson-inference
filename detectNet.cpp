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

inline static bool rectOverlap(const float4& r1, const float4& r2)
{
    return ! ( r2.x > r1.z  
        || r2.z < r1.x
        || r2.y > r1.w
        || r2.w < r1.y
        );
}

static void mergeRect( std::vector<float4>& rects, const float4& rect )
{
	const uint32_t num_rects = rects.size();
	
	bool intersects = false;
	
	for( uint32_t r=0; r < num_rects; r++ )
	{
		if( rectOverlap(rect, rects[r]) )
			intersects = true;
	} 
	
	if( !intersects )
		rects.push_back(rect);
}
	
	
// Detect
int detectNet::Detect( float* rgba, uint32_t width, uint32_t height, float* confidence )
{
	if( !rgba || width == 0 || height == 0 )
	{
		printf("detectNet::Detect( 0x%p, %u, %u ) -> invalid parameters\n", rgba, width, height);
		return -1;
	}

	
	// downsample and convert to band-sequential BGR
	if( CUDA_FAILED(cudaPreImageNet((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
									make_float3(104.0069879317889f, 116.66876761696767f, 122.6789143406786f))) )
	{
		printf("detectNet::Classify() -- cudaPreImageNet failed\n");
		return -1;
	}
	
	// process with GIE
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[OUTPUT_CVG].CUDA, mOutputs[OUTPUT_BBOX].CUDA };
	
	mContext->execute(1, inferenceBuffers);

	// cluster detection bboxes
	float* net_cvg   = mOutputs[OUTPUT_CVG].CPU;
	float* net_rects = mOutputs[OUTPUT_BBOX].CPU;
	
	const int ow  = mOutputs[OUTPUT_BBOX].dims.w;
	const int oh  = mOutputs[OUTPUT_BBOX].dims.h;
	const int owh = ow * oh;
	
	const float cell_width  = /*width*/ mInputDims.w / ow;
	const float cell_height = /*height*/ mInputDims.h / oh;
	
	const float scale_x = float(width) / float(mInputDims.w);
	const float scale_y = float(height) / float(mInputDims.h);
	
	printf("cell width %f  height %f\n", cell_width, cell_height);
	printf("scale x %f  y %f\n", scale_x, scale_y);
	
	std::vector<float4> rects;
	rects.reserve(owh);
	
	for( uint32_t y=0; y < oh; y++ )
	{
		for( uint32_t x=0; x < ow; x++)
		{
			const float coverage = net_cvg[y * ow + x];
			
			if( coverage > mCoverageThreshold )
			{
				const float mx = x * cell_width;
				const float my = y * cell_height;
				
				const float x1 = (net_rects[0 * owh + y * ow + x] + mx) * scale_x;		// left
				const float y1 = (net_rects[1 * owh + y * ow + x] + my) * scale_y;		// top
				const float x2 = (net_rects[2 * owh + y * ow + x] + mx) * scale_x;		// right
				const float y2 = (net_rects[3 * owh + y * ow + x] + my) * scale_y;		// bottom
				
				printf("rect x=%u y=%u  cvg=%f  %f %f   %f %f\n", x, y, coverage, x1, x2, y1, y2);
				
				mergeRect( rects, make_float4(x1, y1, x2, y2) );
			}
		}
	}
	
	return 0;
}

