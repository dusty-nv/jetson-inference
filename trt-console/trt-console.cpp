/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "tensorNet.h"

#include "loadImage.h"
#include "cudaUtility.h"

#include <opencv2/calib3d.hpp>
#include <sys/time.h>




template<typename T>
void mat33_inverse( const T m[3][3], T inv[3][3] )
{
	// determinant
	const T det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
			  - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
			  + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

	// inverse
	inv[0][0] = + (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
	inv[0][1] = - (m[0][1] * m[2][2] - m[0][2] * m[2][1]);
	inv[0][2] = + (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
	inv[1][0] = - (m[1][0] * m[2][2] - m[1][2] * m[2][0]);
	inv[1][1] = + (m[0][0] * m[2][2] - m[0][2] * m[2][0]);
	inv[1][2] = - (m[0][0] * m[1][2] - m[0][2] * m[1][0]);
	inv[2][0] = + (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
	inv[2][1] = - (m[0][0] * m[2][1] - m[0][1] * m[2][0]);
	inv[2][2] = + (m[0][0] * m[1][1] - m[0][1] * m[1][0]);

	// scale by determinant
	for( uint32_t i=0; i < 3; i++ )
		for( uint32_t k=0; k < 3; k++ )
			inv[i][k] /= det;
}


template<typename T>
void mat33_print( const T m[3][3], const char* name=NULL )
{
	if( name != NULL )
		printf("%s = \n");

	printf(" [ ");

	for( uint32_t i=0; i < 3; i++ )
	{
		for( uint32_t k=0; k < 3; k++ )
			printf("%f ", m[i][k]);

		if( i < 2 )
			printf("\n   ");
		else
			printf("]\n");
	}
}

	



uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}



cudaError_t cudaPreHomographyNet( float4* inputA, float4* inputB, size_t inputWidth, size_t inputHeight,
				         	    float* output, size_t outputWidth, size_t outputHeight,
					         cudaStream_t stream );

class homographyNet : public tensorNet
{
public:
	static homographyNet* Create()
	{
		homographyNet* net = new homographyNet();

		if( !net->LoadNetwork(NULL, "networks/deep-homography-coco/deep_homography.onnx",
						  NULL, "input_0", "output_0", 1 /*MAX_BATCH_SIZE_DEFAULT*/,
						  TYPE_FP32, DEVICE_GPU) )
		{
			printf(LOG_TRT "failed to load homographyNet\n");
			delete net;
			return NULL;
		}

		return net;
	}

	~homographyNet()
	{

	}

	bool Process( float* imageA, float* imageB, uint32_t width, uint32_t height )
	{
		if( !imageA || !imageB || width == 0 || height == 0 )
		{
			printf(LOG_TRT "homographyNet::Process() -- invalid user inputs\n");
			return false;
		}

		//printf("user input width=%u height=%u\n", width, height);
		//printf("homg input width=%u height=%u\n", mWidth, mHeight);

		if( CUDA_FAILED(cudaPreHomographyNet((float4*)imageA, (float4*)imageB, width, height,
									  mInputCUDA, mWidth, mHeight, GetStream())) )
		{
			printf(LOG_TRT "homographyNet::Process() -- cudaPreHomographyNet() failed\n");
			return false;
		}

		void* bindBuffers[] = { mInputCUDA, mOutputs[0].CUDA };	

		if( !mContext->execute(1, bindBuffers) )
		{
			printf(LOG_TRT "homographyNet::Process() -- failed to execute TensorRT network\n");
			return false;
		}

		PROFILER_REPORT();

		const uint32_t numOutputs = DIMS_C(mOutputs[0].dims);

		for( uint32_t n=0; n < numOutputs; n++ )
			printf("%f ", mOutputs[0].CPU[n]);

		printf("\n");

		const float scale = 32.0f;

		for( uint32_t n=0; n < numOutputs; n++ )
			mOutputs[0].CPU[n] *= scale;

		std::vector<cv::Point2f> pts1;
		std::vector<cv::Point2f> pts2;

		pts1.resize(4);
		pts2.resize(4);

		pts1[0].x = 0.0f;    pts1[0].y = 0.0f;
		pts1[1].x = mWidth;  pts1[1].y = 0.0f;
		pts1[2].x = mWidth;  pts1[2].y = mHeight;
		pts1[3].x = 0.0f;    pts1[3].y = mHeight;

		for( uint32_t n=0; n < 4; n++ )
		{
			pts2[n].x = pts1[n].x + mOutputs[0].CPU[n*2+0];
			pts2[n].y = pts1[n].y + mOutputs[0].CPU[n*2+1];
		}

		for( uint32_t n=0; n < 4; n++ )
			printf("pts1[%u]  x=%f  y=%f\n", n, pts1[n].x, pts1[n].y);

		for( uint32_t n=0; n < 4; n++ )
			printf("pts2[%u]  x=%f  y=%f\n", n, pts2[n].x, pts2[n].y);

		printf("trt-console:  beginning cv::findHomography (%zu)\n", current_timestamp());

		cv::Mat H_cv = cv::findHomography(pts1, pts2);
 
		printf("trt-console:  finished  cv::findHomography (%zu)\n", current_timestamp());

		if( H_cv.cols * H_cv.rows != 9 )
		{
			printf("homographyNet::Process() -- OpenCV matrix is unexpected size (%ix%i)\n", H_cv.cols, H_cv.rows);
			return false;
		}

		double* H_ptr = H_cv.ptr<double>();

		for( uint32_t n=0; n < 9; n++ )
			printf("H[%u] = %f\n", n, H_ptr[n]);

		double H[3][3];
		double H_inv[3][3];

		for( uint32_t i=0; i < 3; i++ )
			for( uint32_t k=0; k < 3; k++ )
				H[i][k] = H_ptr[i*3+k];

		mat33_print(H, "H");
		mat33_inverse(H, H_inv);
		mat33_print(H_inv, "H_inv");
		
		return true;
	}
	
	
protected:
	homographyNet()
	{

	}

};


// main entry point
int main( int argc, char** argv )
{
	printf("trt-console\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
	
	// retrieve filename arguments
	if( argc < 3 )
	{
		printf("trt-console:   two input image filenames required\n");
		return 0;
	}
	

	// load images
	const char* imgFilename[] = { argv[1], argv[2] };

	float* imgCPU[] = { NULL, NULL };
	float* imgCUDA[] = { NULL, NULL };
	int    imgWidth[] = { 0, 0 };
	int    imgHeight[] = { 0, 0 };

	for( uint32_t n=0; n < 2; n++ )
	{
		if( !loadImageRGBA(imgFilename[n], (float4**)&imgCPU[n], (float4**)&imgCUDA[n], &imgWidth[n], &imgHeight[n]) )
		{
			printf("failed to load image #%u '%s'\n", n, imgFilename[n]);
			return 0;
		}
	}

	if( imgWidth[0] != imgWidth[1] || imgHeight[0] != imgHeight[1] )
	{
		printf("the two images must have the same dimensions\n");
		return 0;
	}


	// create homography network
	homographyNet* net = homographyNet::Create();

	if( !net )
	{
		printf("trt-console:  failed to load homographyNet\n");
		return 0;
	}

	net->EnableProfiler();

	// process the network
	for( int n=0; n < 10; n++ )
	{
		if( !net->Process(imgCUDA[0], imgCUDA[1], imgWidth[0], imgHeight[0]) )
		{
			printf("trt-console:  failed to process homographyNet\n");
			return 0;
		}
	}

	delete net;
	return 0;
}
