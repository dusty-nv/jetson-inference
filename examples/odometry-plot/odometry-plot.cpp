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


// main entry point
int main( int argc, char** argv )
{
	//csvRow row("abc 123 456.789");

	//const float f = (float)row(1);

	//printf("columns:  %zu\n", row.Size());
	//printf("column 1:  %f\n", row.Data<float>(1));

	const std::vector<csvData> row = csvData::Parse("abc 123 456.789");

	const char* ch = row[2];
	std::string str2 = row[2];
	float x = row[1].toFloat();

	printf("columns:  %zu\n", row.size());
	printf("column 0:  %f\n", (float)row[0]);
	printf("column 1:  %f\n", (float)row[1]);
	printf("column 2:  %f\n", (float)row[2]);
	printf("column 2 str:  '%s'\n", (const char*)row[2]);

	csvData t = 10.123f;
	printf("csvData t:  %f\n", float(t));

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

	double* pose = (double*)malloc(numOutputs * sizeof(double));
	memset(pose, 0, numOutputs * sizeof(double));

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

			// calculate the error			
			double imgError = 0.0;
			printf("%04i ", (int)nextLine[0]);

			for( int n=0; n < numOutputs; n++ )
			{
				const double net_data = double(net->GetOutput(n));
				const double gt_delta = double(nextLine[n+1]) - double(prevLine[n+1]);				
				const double gt_error = gt_delta - net_data;
				
				imgError += fabs(gt_error);
				pose[n] += net_data;

				printf("%+.6lf [err=%+.6lf] %+.6lf [err=%+.6lf] ", pose[n], pose[n] - double(nextLine[n+1]), net_data, gt_error);
			}

			printf("**error:  %.6lf\n", imgError);

			avgError += imgError;
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
	printf("**per-DoF error:   %lf\n", avgError / (double)numOutputs);

	printf("**computed pose:  ");

	for( int n=0; n < numOutputs; n++ )
		printf("%+.6lf ", pose[n]);

	printf("\n**expected pose:  ");

	for( int n=0; n < numOutputs; n++ )
		printf("%+.6lf ", (double)prevLine[n+1]);

	printf("\n**pose error:     ");

	for( int n=0; n < numOutputs; n++ )
		printf("%+.6lf ", pose[n] - (double)prevLine[n+1]);

	/*
	 * destroy resources
	 */
	printf("\nodometry-plot:  shutting down...\n");

	SAFE_DELETE(net);

	printf("odometry-plot:  shutdown complete\n");
	return 0;
}


