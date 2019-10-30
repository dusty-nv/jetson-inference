/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#include "homographyNet.h"

#include "commandLine.h"
#include "cudaUtility.h"

#include "mat33.h"

#ifdef HAS_HOMOGRAPHY_NET
#include <opencv2/calib3d.hpp>
#endif


#define DEBUG_HOMOGRAPHY

//-------------------------------------------------------------------------------------
#ifdef HAS_HOMOGRAPHY_NET
namespace cv
{
    Mat filterHomographyDecomp(InputArrayOfArrays rotations,
                               InputArrayOfArrays normals,
                               InputArray _beforeRectifiedPoints,
                               InputArray _afterRectifiedPoints,
                               InputArray _pointsMask)
    {
        CV_Assert(_beforeRectifiedPoints.type() == CV_32FC2 && _afterRectifiedPoints.type() == CV_32FC2 && (_pointsMask.empty() || _pointsMask.type() == CV_8U));

        Mat beforeRectifiedPoints = _beforeRectifiedPoints.getMat(), afterRectifiedPoints = _afterRectifiedPoints.getMat(), pointsMask = _pointsMask.getMat();

        Mat possibleSolutions;

        for (int solutionIdx = 0; solutionIdx < rotations.size().area(); solutionIdx++)
        {
            bool solutionValid = true;

            for (int pointIdx = 0; pointIdx < beforeRectifiedPoints.size().area(); pointIdx++) 
		  {
                if (pointsMask.empty() || pointsMask.at<bool>(pointIdx))
                {
                    Mat tempAddMat = Mat(1, 1, CV_64F, double(1));

                    Mat tempPrevPointMat = Mat(beforeRectifiedPoints.at<Point2f>(pointIdx));
                    tempPrevPointMat.convertTo(tempPrevPointMat, CV_64F);
                    tempPrevPointMat.push_back(tempAddMat);

                    Mat tempCurrPointMat = Mat(afterRectifiedPoints.at<Point2f>(pointIdx));
                    tempCurrPointMat.convertTo(tempCurrPointMat, CV_64F);
                    tempCurrPointMat.push_back(tempAddMat);

                    double prevNormDot = tempPrevPointMat.dot(normals.getMat(solutionIdx));
                    double currNormDot = tempCurrPointMat.dot(rotations.getMat(solutionIdx) * normals.getMat(solutionIdx));

                    if (prevNormDot <= 0 || currNormDot <= 0)
                    {
				    printf("invalid solution %i  (point=%i)\n", solutionIdx, pointIdx);
                        solutionValid = false;
                        break;
                    }
                }
            }
            if (solutionValid)
            {
                possibleSolutions.push_back(solutionIdx);
            }
        }

        return possibleSolutions;
    }
}
#endif
//-------------------------------------------------------------------------------------

// constructor
homographyNet::homographyNet() : tensorNet()
{

}


// destructor
homographyNet::~homographyNet()
{

}


// NetworkTypeFromStr
homographyNet::NetworkType homographyNet::NetworkTypeFromStr( const char* modelName )
{
	if( !modelName )
		return homographyNet::CUSTOM;

	homographyNet::NetworkType type = homographyNet::CUSTOM;

	if( strcasecmp(modelName, "coco") == 0 || strcasecmp(modelName, "coco_128") == 0 || strcasecmp(modelName, "coco-128") == 0 )
		type = homographyNet::COCO_128;
	else if( strcasecmp(modelName, "webcam") == 0 || strcasecmp(modelName, "webcam_320") == 0 || strcasecmp(modelName, "webcam-320") == 0 )
		type = homographyNet::WEBCAM_320;
	else
		type = homographyNet::CUSTOM;

	return type;
}


// Create
homographyNet* homographyNet::Create( homographyNet::NetworkType networkType, uint32_t maxBatchSize, 
					   precisionType precision, deviceType device, bool allowGPUFallback )
{
#ifndef HAS_HOMOGRAPHY_NET
	printf(LOG_TRT "error -- homographyNet is supported only in TensorRT 5.0 and newer\n");
	return NULL;
#endif

	if( networkType == COCO_128 )
		return Create("networks/Deep-Homography-COCO/deep_homography.onnx", HOMOGRAPHY_NET_DEFAULT_INPUT, HOMOGRAPHY_NET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else if( networkType == WEBCAM_320 )
		return Create("networks/Deep-Homography-Webcam-320/deep_homography_webcam_320.onnx", HOMOGRAPHY_NET_DEFAULT_INPUT, HOMOGRAPHY_NET_DEFAULT_OUTPUT, maxBatchSize, precision, device, allowGPUFallback);
	else
		return NULL;
}


// Create
homographyNet* homographyNet::Create( const char* model_path, const char* input, 
							   const char* output, uint32_t maxBatchSize,
					   		   precisionType precision, deviceType device, 
						        bool allowGPUFallback )
{
#ifndef HAS_HOMOGRAPHY_NET
	printf(LOG_TRT "error -- homographyNet is supported only in TensorRT 5.0 and newer\n");
	return NULL;
#endif

	if( !model_path || !input || !output )
		return NULL;

	printf("\n");
	printf("homographyNet -- loading homography network model from:\n");
	printf("         -- model        %s\n", model_path);
	printf("         -- input_blob   '%s'\n", input);
	printf("         -- output_blob  '%s'\n", output);
	printf("         -- batch_size   %u\n\n", maxBatchSize);

	// create the homography network
	homographyNet* net = new homographyNet();
	
	if( !net )
		return NULL;
	
	// load the model
	if( !net->LoadNetwork(NULL, model_path, NULL,
					  input, output, maxBatchSize,
					  precision, device, allowGPUFallback) )
	{
		printf(LOG_TRT "failed to load homographyNet\n");
		delete net;
		return NULL;
	}
	
	printf(LOG_TRT "%s loaded\n", model_path);
	return net;
}


// Create
homographyNet* homographyNet::Create( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);

	const char* model = cmdLine.GetString("network");

	if( !model )
	{
		model = cmdLine.GetString("model");

		if( !model )
			return homographyNet::Create();
	}

	homographyNet::NetworkType type = NetworkTypeFromStr(model);

	if( type == homographyNet::CUSTOM )
	{
		const char* input    = cmdLine.GetString("input_blob");
		const char* output   = cmdLine.GetString("output_blob");

		if( !input )  input  = HOMOGRAPHY_NET_DEFAULT_INPUT;
		if( !output ) output = HOMOGRAPHY_NET_DEFAULT_OUTPUT;
		
		int maxBatchSize = cmdLine.GetInt("batch_size");
		
		if( maxBatchSize < 1 )
			maxBatchSize = 1;

		return homographyNet::Create(model, input, output, maxBatchSize);
	}

	// create from pretrained model
	return homographyNet::Create(type);
}


// from homographyNet.cu
cudaError_t cudaPreHomographyNet( float4* inputA, float4* inputB, size_t inputWidth, size_t inputHeight,
				         	    float* output, size_t outputWidth, size_t outputHeight,
					         cudaStream_t stream );


// FindDisplacement
bool homographyNet::FindDisplacement( float* imageA, float* imageB, uint32_t width, uint32_t height, float displacement[8] )
{
#ifdef HAS_HOMOGRAPHY_NET
	if( !imageA || !imageB || width == 0 || height == 0 )
	{
		printf(LOG_TRT "homographyNet::Process() -- invalid user inputs\n");
		return false;
	}

	//printf("user input width=%u height=%u\n", width, height);
	//printf("homg input width=%u height=%u\n", mWidth, mHeight);

	PROFILER_BEGIN(PROFILER_PREPROCESS);

	/*
	 * convert/rescale the individual RGBA images into grayscale planar format
	 */
	if( CUDA_FAILED(cudaPreHomographyNet((float4*)imageA, (float4*)imageB, width, height,
								  mInputs[0].CUDA, GetInputWidth(), GetInputHeight(), 
								  GetStream())) )
	{
		printf(LOG_TRT "homographyNet::Process() -- cudaPreHomographyNet() failed\n");
		return false;
	}

	PROFILER_END(PROFILER_PREPROCESS);
	PROFILER_BEGIN(PROFILER_NETWORK);

	/*
	 * perform the inferencing
 	 */
	if( !ProcessNetwork() )
		return false;

	PROFILER_END(PROFILER_NETWORK);

	const uint32_t numOutputs = DIMS_C(mOutputs[0].dims);

#ifdef DEBUG_HOMOGRAPHY
	printf("raw " );

	for( uint32_t n=0; n < numOutputs; n++ )
		printf("%f ", mOutputs[0].CPU[n]);

	printf("\n");
#endif

	/*
	 * rescale the raw outputs
	 */
	const float scale = 32.0f;

	for( uint32_t n=0; n < numOutputs; n++ )
		displacement[n] = mOutputs[0].CPU[n] * scale;

#ifdef DEBUG_HOMOGRAPHY
	printf("*32 " );

	for( uint32_t n=0; n < numOutputs; n++ )
		printf("%f ", displacement[n]);

	printf("\n");
#endif

	return true;
#else
	printf(LOG_TRT "error -- homographyNet is supported only in TensorRT 5.0 and newer\n");
	return false;
#endif
}
	

// ComputeHomography
bool homographyNet::ComputeHomography( const float displacement[8], float H[3][3], float H_inv[3][3] )
{
#ifdef HAS_HOMOGRAPHY_NET
	PROFILER_BEGIN(PROFILER_POSTPROCESS);

	const uint32_t width = GetInputWidth();
	const uint32_t height = GetInputHeight();

	/*
	 * translate the x/y displacements back into corner points
	 */
	std::vector<cv::Point2f> pts1;
	std::vector<cv::Point2f> pts2;

	pts1.resize(4);
	pts2.resize(4);

	pts1[0].x = 0.0f;   pts1[0].y = 0.0f;
	pts1[1].x = width;  pts1[1].y = 0.0f;
	pts1[2].x = width;  pts1[2].y = height;
	pts1[3].x = 0.0f;   pts1[3].y = height;

	for( uint32_t n=0; n < 4; n++ )
	{
		pts2[n].x = pts1[n].x + displacement[n*2+0];
		pts2[n].y = pts1[n].y + displacement[n*2+1];
	}

#ifdef DEBUG_HOMOGRAPHY
	for( uint32_t n=0; n < 4; n++ )
		printf("pts1[%u]  x=%f  y=%f\n", n, pts1[n].x, pts1[n].y);

	for( uint32_t n=0; n < 4; n++ )
		printf("pts2[%u]  x=%f  y=%f\n", n, pts2[n].x, pts2[n].y);
#endif

	/*
	 * estimate the homography using DLT
	 */
	cv::Mat H_cv = cv::findHomography(pts1, pts2);

	if( H_cv.cols * H_cv.rows != 9 )
	{
		printf("homographyNet::Process() -- OpenCV matrix is unexpected size (%ix%i)\n", H_cv.cols, H_cv.rows);
		return false;
	}


	/*
	 * compute the homography's inverse
	 */
	double* H_ptr = H_cv.ptr<double>();

	//double H[3][3];
	//double H_inv[3][3];

	for( uint32_t i=0; i < 3; i++ )
		for( uint32_t k=0; k < 3; k++ )
			H[i][k] = H_ptr[i*3+k];

	mat33_inverse(H_inv, H);

#ifdef DEBUG_HOMOGRAPHY
	mat33_print(H, "H");	
	mat33_print(H_inv, "H_inv");
#endif

	PROFILER_END(PROFILER_POSTPROCESS);
	//PROFILER_REPORT();

	return true;
#else
	printf(LOG_TRT "error -- homographyNet is supported only in TensorRT 5.0 and newer\n");
	return false;
#endif
}


// ComputeHomography
bool homographyNet::ComputeHomography( const float displacement[8], float H[3][3] )
{
	float H_inv[3][3];
	return ComputeHomography(displacement, H, H_inv);
}

	
// FindHomography
bool homographyNet::FindHomography( float* imageA, float* imageB, uint32_t width, uint32_t height, float H[3][3], float H_inv[3][3] )
{
	float displacement[8];

	if( !FindDisplacement(imageA, imageB, width, height, displacement) )
		return false;

	return ComputeHomography(displacement, H, H_inv);
}


// FindHomography
bool homographyNet::FindHomography( float* imageA, float* imageB, uint32_t width, uint32_t height, float H[3][3] )
{
	float H_inv[3][3];
	return FindHomography(imageA, imageB, width, height, H, H_inv);
}


	
#if 0
	/*
	 * create a default intrinsic camera calibration matrix
	 * note:  should use a real calibration matrix here
	 */
	cv::Mat cam_intrinsic = cv::Mat::zeros(3, 3, CV_64FC1);	// CV_32FC1

	// focal length  (TODO: fix for image size != 128)
	const double fx = 114.0;		// F = (img_size/2) * tan(FoV/2)
	const double fy = fx;		// F = (128/2) * tan(45/2)	

	cam_intrinsic.at<double>(0,0) = fx;
	cam_intrinsic.at<double>(1,1) = fy;
	cam_intrinsic.at<double>(2,2) = 1.0;

	cam_intrinsic.at<double>(0,2) = double(mWidth - 1) * 0.5;
	cam_intrinsic.at<double>(1,2) = double(mHeight - 1) * 0.5;

	
	/*
	 * decompose the homography
	 */
	std::vector<cv::Mat> Rs_decomp, ts_decomp, normals_decomp;

	printf("trt-console:  beginning cv::decomposeHomography (%zu)\n", current_timestamp());
	const int solutions = cv::decomposeHomographyMat(H_cv, cam_intrinsic, Rs_decomp, ts_decomp, normals_decomp);
	printf("trt-console:  finished  cv::decomposeHomography (%zu)\n", current_timestamp());
	
	std::cout << std::endl << "Decompose homography matrix computed from the camera displacement:" << std::endl;
	
	for (int i = 0; i < solutions; i++)
	{
		const double factor_d1 = 1.0; //const double factor_d1 = 1.0 / d_inv1;
		
		cv::Mat rvec_decomp;
		cv::Rodrigues(Rs_decomp[i], rvec_decomp);

		std::cout << std::endl << "Solution " << i << ":" << std::endl;
		std::cout << "rvec from homography decomposition: " << rvec_decomp.t() << std::endl;
		//std::cout << "rvec from camera displacement: " << rvec_1to2.t() << std::endl;
		std::cout << "tvec from homography decomposition: " << ts_decomp[i].t() << " and scaled by d: " << factor_d1 * ts_decomp[i].t() << std::endl;
		//std::cout << "tvec from camera displacement: " << t_1to2.t() << std::endl;
		std::cout << "plane normal from homography decomposition: " << normals_decomp[i].t() << std::endl;
		//std::cout << "plane normal at camera 1 pose: " << normal1.t() << std::endl << std::endl;
	}


	/*
	 * filter the possible decomposition solutions
	 */
	cv::Mat filtered_decomp = cv::filterHomographyDecomp(Rs_decomp, normals_decomp,
											   pts1, pts2, cv::Mat());
	
	printf("filtered solutions mat (%ix%i) (type=%i)\n", filtered_decomp.cols, filtered_decomp.rows, filtered_decomp.type());
#endif
			 

