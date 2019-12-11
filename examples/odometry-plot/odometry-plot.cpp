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

#include "odometryNet.h"

#include "cudaMappedMemory.h"
#include "cudaWarp.h"

#include "loadImage.h"
#include "commandLine.h"
#include "mat33.h"

#include "csvReader.h"
#include "csvWriter.h"

const int DOF = 3;
const char* DOF_names[] = { "x", "y", "θ" };


// print usage
int print_usage()
{
	printf("usage: odometry-plot [-h] [--network NETWORK]\n");
	printf("                     file_A file_B\n\n");
	printf("Perform visual odometry estimation on a sequential pair of images\n\n");
	printf("positional arguments:\n");
	printf("  file_A               filename of the first input image to process\n");
	printf("  file_B               filename of the second input image to process\n\n");
	printf("optional arguments:\n");
	printf("  --help               show this help message and exit\n");
	printf("  --network NETWORK    pre-trained model to load (see below for options)\n");
	printf("%s\n", odometryNet::Usage());

	return 0;
}


void printImg( float* img, int width, int height )
{
	const int n = width * height;

	for( int c=0; c < 3; c++ )
	{
		printf("[");

		for( int y=0; y < 3; y++ )
		{
			int idx = c * n + y * width; 
			printf("[%.4f, %.4f, %.4f, ..., ", img[idx+0], img[idx+1], img[idx+2]);
			idx += (width - 3);
			printf("%.4f, %.4f, %.4f]\n", img[idx+0], img[idx+1], img[idx+2]);
		}

		printf("...,\n");

		for( int y=height-3; y < height; y++ )
		{
			int idx = c * n + y * width; 
			printf("[%.4f, %.4f, %.4f, ..., ", img[idx+0], img[idx+1], img[idx+2]);
			idx += (width - 3);
			printf("%.4f, %.4f, %.4f]\n", img[idx+0], img[idx+1], img[idx+2]);
		}
	
		printf("]\n");
	}
}


// main entry point
int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	const char* imgPath = cmdLine.GetPosition(0);
	
	if( !imgPath )
	{
		printf("odometry-plot:   path to input image directory required\n");
		return print_usage();
	}

	// load CSV
	csvReader* csv = csvReader::Open(cmdLine.GetString("gt"));

	if( !csv )
	{
		printf("odometry-plot:  failed to open groundtruth CSV file\n");
		return 0;
	}

	csvWriter* csvResults = csvWriter::Open(cmdLine.GetString("results-out"), " ");
	csvWriter* csvResultsGT = csvWriter::Open(cmdLine.GetString("gt-out"), " ");

#if 0
		printf("opened csv %s\n", csv->GetFilename());

		std::vector<csvData> line;

		while( csv->Read(line) )
		{
			const int numColumns = line.size();

			for( int n=0; n < numColumns; n++ )
				printf("['%s' %f] ", (const char*)line[n], (float)line[n]);
		
			printf("\n");
		}
#endif

	/*
	 * load network
	 */
	odometryNet* net = odometryNet::Create(argc, argv);

	if( !net )
	{
		printf("odometry-plot:  failed to load network\n");
		return 0;
	}

	/*
	 * allocate storage for pose/location
	 */
	const int numOutputs = net->GetNumOutputs();

	//double* pose = (double*)malloc(numOutputs * sizeof(double));
	//memset(pose, 0, numOutputs * sizeof(double));

	double pose[DOF];
	memset(pose, 0, sizeof(pose));


	/* 
	 * process image sequence
	 */
	int imgWidth = 0;
	int imgHeight = 0;

	float4* imgPrev = NULL;
	float4* imgNext = NULL;

	std::vector<csvData> prevLine;
	std::vector<csvData> nextLine;

	double avgError = 0.0;
	uint32_t numImages = 0;

	while( csv->Read(nextLine) )
	{
		// verify the CSV row has enough data
		if( numOutputs >= nextLine.size() )
		{
			printf("odometry-plot:  frame %s did not have at least %i CSV groundtruth data\n", (const char*)nextLine[0], numOutputs);
			break;
		} 

		// load the next frame
		char imgFilename[512];
		sprintf(imgFilename, "%s/%04i.ppm", imgPath, (int)nextLine[0]);
	
		if( !loadImageRGBA(imgFilename, &imgNext, &imgWidth, &imgHeight) )
			break;

		// process the next frame
		if( imgPrev != NULL && imgNext != NULL )
		{
	 		// estimate the odometry with the network
			if( !net->Process(imgPrev, imgNext, imgWidth, imgHeight) )
			{
				printf("odometry-plot:  failed to find odometry\n");
				break;
			}

			// print out performance info
			//net->PrintProfilerTimes();

			// accumulate the pose
			const double velocity = net->GetOutput(0);
			const double delta_heading = net->GetOutput(1);

			pose[2] += delta_heading;

			const double delta[] = { velocity * cos(pose[2]),  // dx
								velocity * sin(pose[2]),  // dy
								delta_heading };

			pose[0] += delta[0];
			pose[1] += delta[1];

			// calculate the error			
			double mseError = 0.0;
			printf("%04i ", (int)nextLine[0]);

#if 1
			for( int n=0; n < DOF; n++ )
			{
				const double gt_value = double(nextLine[n+1]);
				const double gt_delta = gt_value - double(prevLine[n+1]);
				const double gt_error = pose[n] - gt_value;

				mseError += gt_error * gt_error;

				printf("%s=[%+.6lf (gt=%+.6lf) (err=%+.6lf) Δ %+.6lf (gt=%+.6lf) (err=%+.6lf)] ", DOF_names[n], pose[n], gt_value, gt_error, delta[n], gt_delta, delta[n] - gt_delta);
			}

			const double gt_next_x   = double(nextLine[1]);
			const double gt_next_y   = double(nextLine[2]);
			const double gt_delta_x  = gt_next_x - double(prevLine[1]);
			const double gt_delta_y  = gt_next_y - double(prevLine[2]);
			const double gt_velocity = sqrt(gt_delta_x * gt_delta_x + gt_delta_y * gt_delta_y);

			printf("v=[%+.6lf (gt=%+.6lf) (err=%+.6lf)] ", velocity, gt_velocity, velocity - gt_velocity);

			mseError /= double(DOF);	
			
			// output the results
			if( csvResults != NULL )
				csvResults->WriteLine(pose[0], pose[1]);

			if( csvResultsGT != NULL )
				csvResultsGT->WriteLine(gt_next_x, gt_next_y);		
#else
			for( int n=0; n < numOutputs; n++ )
			{
				const double net_data = double(net->GetOutput(n));
				const double gt_delta = double(nextLine[n+1]) - double(prevLine[n+1]);				
				const double gt_error = net_data - gt_delta;
				
				mseError += gt_error * gt_error; //fabs(gt_error);
				pose[n] += net_data;

				printf("%+.6lf [err=%+.6lf] %+.6lf [err=%+.6lf] ", pose[n], pose[n] - double(nextLine[n+1]), net_data, gt_error);
			}

			mseError /= double(numOutputs);
#endif						
			printf("**error:  %.6lf\n", mseError);

			//printf("%04i ", (int)nextLine[0]);
			//printImg(net->GetInput(), net->GetInputWidth(), net->GetInputHeight());

			avgError += mseError;
			numImages++;
		}

		// cleanup the frame
		if( imgPrev != NULL )
			CUDA(cudaFreeHost(imgPrev));

		imgPrev  = imgNext;
		prevLine = nextLine;
	}

	avgError /= (double)numImages;

	printf("**average error:   %lf\n", avgError);
	printf("**computed pose:  ");

	for( int n=0; n < DOF; n++ )
		printf("%s = %+.6lf ", DOF_names[n], pose[n]);

	printf("\n**expected pose:  ");

	for( int n=0; n < DOF; n++ )
		printf("%s = %+.6lf ", DOF_names[n], (double)prevLine[n+1]);

	printf("\n**pose error:     ");

	for( int n=0; n < DOF; n++ )
		printf("%s = %+.6lf ", DOF_names[n], pose[n] - (double)prevLine[n+1]);

	/*
	 * destroy resources
	 */
	printf("\nodometry-plot:  shutting down...\n");

	SAFE_DELETE(csv);
	SAFE_DELETE(csvResults);
	SAFE_DELETE(csvResultsGT);

	SAFE_DELETE(net);

	printf("odometry-plot:  shutdown complete\n");
	return 0;
}


