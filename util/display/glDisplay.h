/*
 * inference-101
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

