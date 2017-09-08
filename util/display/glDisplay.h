/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __GL_VIEWPORT_H__
#define __GL_VIEWPORT_H__


#include "glUtility.h"
#include "glTexture.h"

#include <time.h>


/**
 * OpenGL display window / video viewer
 */
class glDisplay
{
public:
	/**
	 * Create a new maximized openGL display window.
	 */
	static glDisplay* Create();

	/**
	 * Destroy window
	 */
	~glDisplay();

	/**
 	 * Clear window and begin rendering a frame.
	 */
	void BeginRender();

	/**
	 * Finish rendering and refresh / flip the backbuffer.
	 */
	void EndRender();

	/**
	 * Process UI events.
	 */
	void UserEvents();
		
	/**
	 * UI event handler.
	 */
	void onEvent( uint msg, int a, int b );

	/**
	 * Set the window title string.
	 */
	void SetTitle( const char* str );

	/**
	 * Get the average frame time (in milliseconds).
	 */
	inline float GetFPS()	{ return 1000000000.0f / mAvgTime; }
		
protected:
	glDisplay();
		
	bool initWindow();
	bool initGL();

	static const int screenIdx = 0;
		
	Display*     mDisplayX;
	Screen*      mScreenX;
	XVisualInfo* mVisualX;
	Window       mWindowX;
	GLXContext   mContextGL;
		
	uint32_t mWidth;
	uint32_t mHeight;

	timespec mLastTime;
	float    mAvgTime;
};

#endif

