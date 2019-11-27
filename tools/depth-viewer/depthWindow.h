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

#ifndef __DEPTH_VIEWER_WINDOW__
#define __DEPTH_VIEWER_WINDOW__

#include "commandLine.h"

#include "cudaPointCloud.h"
#include "cudaColormap.h"

#include "depthNet.h"
#include "stereoNet.h"
#include "segNet.h"


// forward declarations
class gstCamera;
class glDisplay;


/*
 * Depth viewer window
 */
class DepthWindow
{
public:
	// create the window and processing objects
	static DepthWindow* Create( commandLine& cmdLine );

	// close the window and camera object
	~DepthWindow();

	// capture & render next camera frame
	bool Render();

	// window open/closed status
	bool IsOpen() const;
	bool IsClosed() const;

	// camera streaming status
	bool IsStreaming() const;

protected:
	DepthWindow();

	bool init( commandLine& cmdLine );
	bool process();

	gstCamera* mCamera;
	glDisplay* mDisplay;

	depthNet*  mDepthNet;
	stereoNet* mStereoNet;
	segNet*	 mSegNet;

	float* mImages[2];
	float* mDepthImg;
	float* mSegOverlay;
	float* mSegMask;

	bool     mNewImages;
	uint32_t mNumImages;
	uint32_t mImgWidth;
	uint32_t mImgHeight;
	
	cudaPointCloud*  mPointCloud;
	cudaColormapType mColormap;
	cudaFilterMode   mFilterMode;
};

#endif

