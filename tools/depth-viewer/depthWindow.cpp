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

#include "depthWindow.h"

#include "gstCamera.h"
#include "glDisplay.h"

#include "cudaMappedMemory.h"
#include "imageIO.h"

#include <locale.h>


// constructor
DepthWindow::DepthWindow()
{
	mCamera  = NULL;
	mDisplay = NULL;

	mDepthNet   = NULL;
	mStereoNet  = NULL;
	mSegNet     = NULL;
	mPointCloud = NULL;

	mImages[0]  = NULL;
	mImages[1]  = NULL;
	mDepthImg   = NULL;
	mSegOverlay = NULL;
	mSegMask    = NULL;

	mNewImages  = false;
	mNumImages  = 0;
	mImgWidth   = 0;
	mImgHeight  = 0;
}


// destructor
DepthWindow::~DepthWindow()
{
	if( mDepthImg != NULL )
	{
		CUDA(cudaFree(mDepthImg));
		mDepthImg = NULL;
	}

	if( mSegOverlay != NULL )
	{
		CUDA(cudaFree(mSegOverlay));
		mSegOverlay = NULL;
	}

	if( mSegMask != NULL )
	{
		CUDA(cudaFree(mSegMask));
		mSegMask = NULL;
	}

	SAFE_DELETE(mDepthNet);
	SAFE_DELETE(mStereoNet);
	SAFE_DELETE(mSegNet);

	SAFE_DELETE(mPointCloud);
	SAFE_DELETE(mCamera);
	SAFE_DELETE(mDisplay);
}


// Create
DepthWindow* DepthWindow::Create( commandLine& cmdLine )
{
	DepthWindow* window = new DepthWindow();

	if( !window || !window->init(cmdLine) )
	{
		printf("depth-viewer:  DepthWindow::Create() failed\n");
		return NULL;
	}

	return window;
}


// init
bool DepthWindow::init( commandLine& cmdLine )
{
	const uint32_t numPositionArgs = cmdLine.GetPositionArgs();

	// either load images from disk, or open camera device
	if( numPositionArgs > 0 )
	{
		for( uint32_t n=0; n < numPositionArgs && n < 2; n++ )
		{
			int imgWidth = 0;
			int imgHeight = 0;

			if( !loadImageRGBA(cmdLine.GetPosition(n), (float4**)&mImages[n], &imgWidth, &imgHeight) )
				return false;

			if( n == 0 )
			{
				mImgWidth = imgWidth;
				mImgHeight = imgHeight;
			}
			else if( imgWidth != mImgWidth || imgHeight != mImgHeight )
			{
				printf("depth-viewer:  image dimensions must match (%ux%u vs %ix%i)\n", mImgWidth, mImgHeight, imgWidth, imgHeight);
				return false;
			}

			mNumImages++;
		}

		mNewImages = true;
	}
	else
	{
		mCamera = gstCamera::Create(cmdLine.GetInt("width", gstCamera::DefaultWidth),
							   cmdLine.GetInt("height", gstCamera::DefaultHeight),
							   cmdLine.GetString("camera"));

		if( !mCamera )
		{
			printf("depth-viewer:  failed to initialize camera device\n");
			return false;
		}

		printf("depth-viewer:  successfully initialized camera device\n");
		printf("    width:  %u\n", mCamera->GetWidth());
		printf("   height:  %u\n", mCamera->GetHeight());
		printf("    depth:  %u (bpp)\n\n", mCamera->GetPixelDepth());
	}

	// either load stereo or mono-depth network
	if( mNumImages > 1 || cmdLine.GetString("stereo") != NULL )
	{
		mStereoNet = stereoNet::Create(stereoNet::NetworkTypeFromStr(cmdLine.GetString("stereo", "resnet18-2d")));

		if( !mStereoNet )
		{
			printf("depth-viewer:  failed to load stereo network\n");
			return false;
		}
	}
	else
	{
		mDepthNet = depthNet::Create(depthNet::NetworkTypeFromStr(cmdLine.GetString("depth", "mobilenet")));

		if( !mDepthNet )
		{
			printf("depth-viewer:  failed to load mono-depth network\n");
			return false;
		}
	}

	// load segmentation network if desired
	const char* segModel = cmdLine.GetString("segmentation");

	if( segModel != NULL )
	{
		mSegNet = segNet::Create(segNet::NetworkTypeFromStr(segModel));

		if( !mSegNet )
			printf("depth-viewer:  failed to load segmentation network\n");
		else
			mSegNet->SetOverlayAlpha(cmdLine.GetFloat("alpha", 120.0f));
	}

	// parse the desired colormap and filter mode
	mColormap = cudaColormapFromStr(cmdLine.GetString("colormap"));
	mFilterMode = cudaFilterModeFromStr(cmdLine.GetString("filter-mode"));

	// create point cloud
	mPointCloud = cudaPointCloud::Create();

	if( !mPointCloud ) 
	{
		printf("depth-viewer:  failed to create point cloud\n");
		return false;
	}

	// load camera calibration
	const char* calibration = cmdLine.GetString("calibration");

	if( calibration != NULL )
		mPointCloud->SetCalibration(calibration);

	// create openGL window
	mDisplay = glDisplay::Create();

	if( !mDisplay ) 
	{
		printf("depth-viewer:  failed to create openGL display\n");
		return false;
	}

	// enable commas in sprintf
	setlocale(LC_NUMERIC, "");

	return true;
}
	
// process
bool DepthWindow::process()
{
	float*   depthField  = NULL;
	uint32_t depthWidth  = 0;
	uint32_t depthHeight = 0;

	// process depth
	if( mDepthNet != NULL )
	{
		if( !mDepthNet->Process(mImages[0], mImgWidth, mImgHeight, 
						    mDepthImg, mImgWidth/2, mImgHeight/2, 
						    mColormap, mFilterMode) )
		{
			printf("depth-viewer:  failed to process mono depth map\n");
			return false;
		}

		depthField  = mDepthNet->GetDepthField();
		depthWidth  = mDepthNet->GetDepthFieldWidth();
		depthHeight = mDepthNet->GetDepthFieldHeight();

		// wait for GPU to complete work			
		CUDA(cudaDeviceSynchronize());

		// print out performance info
		mDepthNet->PrintProfilerTimes();
	}
	else if( mStereoNet != NULL )
	{
		// TODO
	}

	// process segmentation
	if( mSegNet != NULL )
	{
		if( !mSegNet->Process(mImages[0], mImgWidth, mImgHeight) )
		{
			printf("depth-viewer:  failed to process segmentation\n");
			return false;
		}
		
		if( !mSegNet->Overlay(mSegOverlay, mImgWidth, mImgHeight, (segNet::FilterMode)mFilterMode) )
		{
			printf("depth-viewer:  failed to process segmentation overlay.\n");
			return false;
		}

		if( !mSegNet->Mask(mSegMask, mImgWidth/2, mImgHeight/2, (segNet::FilterMode)mFilterMode) )
		{
			printf("depth-viewer:  failed to process segmentation mask.\n");
			return false;
		}

		mImages[0] = mSegOverlay;
	}


	// extract point cloud
	if( !mPointCloud->Extract(depthField, depthWidth, depthHeight, (float4*)mImages[0], mImgWidth, mImgHeight) )
	{
		printf("depth-viewer:  failed to extract point cloud\n");
		return false;
	}

	return true;
}


// Render
bool DepthWindow::Render()
{
	// capture RGBA image
	if( mCamera != NULL )
	{
		float* imgRGBA = NULL;

		if( mCamera->CaptureRGBA(&imgRGBA) )
		{
			mImages[0] = imgRGBA;
			mImgWidth  = mCamera->GetWidth();
			mImgHeight = mCamera->GetHeight();
		}
		else
		{
			printf("depth-viewer:  failed to capture RGBA image from camera\n");
		}
	}

	// allocate depth image
	if( !mDepthImg )
	{
		if( CUDA_FAILED(cudaMalloc(&mDepthImg, mImgWidth/2 * mImgHeight/2 * sizeof(float4))) )
			return false;
	}

	// allocate segmentation images
	if( mSegNet != NULL )
	{
		if( !mSegOverlay )
		{
			if( CUDA_FAILED(cudaMalloc(&mSegOverlay, mImgWidth * mImgHeight * sizeof(float4))) )
				return false;
		}

		if( !mSegMask )
		{
			if( CUDA_FAILED(cudaMalloc(&mSegMask, mImgWidth/2 * mImgHeight/2 * sizeof(float4))) )
				return false;
		}
	}

	// process image(s)
	if( mCamera != NULL || mNewImages )
	{
		if( !process() )
			printf("depth-viewer:  failed to process latest frame\n");

		mNewImages = false;
	}

	// update display
	if( mDisplay != NULL )
	{
		// begin the frame
		mDisplay->BeginRender();

		// render the images
		mDisplay->Render(mImages[0], mImgWidth, mImgHeight);
		mDisplay->Render(mDepthImg, mImgWidth/2, mImgHeight/2, mImgWidth);

		if( mSegNet )
			mDisplay->Render(mSegMask, mImgWidth/2, mImgHeight/2, mImgWidth, mImgHeight/2 + 30);

		// render the point cloud
		mDisplay->SetViewport(0, mImgHeight + 30, mDisplay->GetWidth(), mDisplay->GetHeight());
		mDisplay->RenderRect(0.15f, 0.15f, 0.15f);
		mPointCloud->Render();
		mDisplay->ResetViewport();

		// update the status bar
		char str[256];
		sprintf(str, "Depth Viewer | %'u Points | %.0f FPS", mPointCloud->GetNumPoints(), mDisplay->GetFPS());
		mDisplay->SetTitle(str);

		// present the frame
		mDisplay->EndRender();
	}
}


// IsOpen
bool DepthWindow::IsOpen() const
{
	return mDisplay->IsOpen();
}


// IsClosed
bool DepthWindow::IsClosed() const
{
	return mDisplay->IsClosed();
}


// IsStreaming
bool DepthWindow::IsStreaming() const
{
	if( !mCamera )
		return false;

	return mCamera->IsStreaming();
}


