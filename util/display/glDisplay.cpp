/*
 * inference-101
 */
 
#include "glDisplay.h"


 
// Constructor
glDisplay::glDisplay()
{
	mWindowX   = 0;
	mScreenX   = NULL;
	mVisualX   = NULL;
	mContextGL = NULL;
	mDisplayX  = NULL;
	mWidth     = 0;
	mHeight    = 0;
	mAvgTime   = 1.0f;

	clock_gettime(CLOCK_REALTIME, &mLastTime);
}


// Destructor
glDisplay::~glDisplay()
{
	glXDestroyContext(mDisplayX, mContextGL);
}


// Create
glDisplay* glDisplay::Create()
{
	glDisplay* vp = new glDisplay();
	
	if( !vp )
		return NULL;
		
	if( !vp->initWindow() )
	{
		printf("[OpenGL]  failed to create X11 Window.\n");
		delete vp;
		return NULL;
	}
	
	if( !vp->initGL() )
	{
		printf("[OpenGL]  failed to initialize OpenGL.\n");
		delete vp;
		return NULL;
	}
	
	GLenum err = glewInit();
	
	if (GLEW_OK != err)
	{
		printf("[OpenGL]  GLEW Error: %s\n", glewGetErrorString(err));
		delete vp;
		return NULL;
	}

	printf("[OpenGL]  glDisplay display window initialized\n");
	return vp;
}


// initWindow
bool glDisplay::initWindow()
{
	if( !mDisplayX )
		mDisplayX = XOpenDisplay(0);

	if( !mDisplayX )
	{
		printf( "[OpenGL]  failed to open X11 server connection." );
		return false;
	}

		
	if( !mDisplayX )
	{
		printf( "InitWindow() - no X11 server connection." );
		return false;
	}

	// retrieve screen info
	const int screenIdx   = DefaultScreen(mDisplayX);
	const int screenWidth = DisplayWidth(mDisplayX, screenIdx);
	const int screenHeight = DisplayHeight(mDisplayX, screenIdx);
	
	printf("default X screen %i:   %i x %i\n", screenIdx, screenWidth, screenHeight);
	
	Screen* screen = XScreenOfDisplay(mDisplayX, screenIdx);

	if( !screen )
	{
		printf("failed to retrieve default Screen instance\n");
		return false;
	}
	
	Window winRoot = XRootWindowOfScreen(screen);

	// get framebuffer format
	static int fbAttribs[] =
	{
			GLX_X_RENDERABLE, True,
			GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
			GLX_RENDER_TYPE, GLX_RGBA_BIT,
			GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
			GLX_RED_SIZE, 8,
			GLX_GREEN_SIZE, 8,
			GLX_BLUE_SIZE, 8,
			GLX_ALPHA_SIZE, 8,
			GLX_DEPTH_SIZE, 24,
			GLX_STENCIL_SIZE, 8,
			GLX_DOUBLEBUFFER, True,
			GLX_SAMPLE_BUFFERS, 0,
			GLX_SAMPLES, 0,
			None
	};

	int fbCount = 0;
	GLXFBConfig* fbConfig = glXChooseFBConfig(mDisplayX, screenIdx, fbAttribs, &fbCount);

	if( !fbConfig || fbCount == 0 )
		return false;

	// get a 'visual'
	XVisualInfo* visual = glXGetVisualFromFBConfig(mDisplayX, fbConfig[0]);

	if( !visual )
		return false;

	// populate windows attributes
	XSetWindowAttributes winAttr;
	winAttr.colormap = XCreateColormap(mDisplayX, winRoot, visual->visual, AllocNone);
	winAttr.background_pixmap = None;
	winAttr.border_pixel = 0;
	winAttr.event_mask = StructureNotifyMask|KeyPressMask|KeyReleaseMask|PointerMotionMask|ButtonPressMask|ButtonReleaseMask;

	
	// create window
	Window win = XCreateWindow(mDisplayX, winRoot, 0, 0, screenWidth, screenHeight, 0,
							   visual->depth, InputOutput, visual->visual, CWBorderPixel|CWColormap|CWEventMask, &winAttr);

	if( !win )
		return false;

	XStoreName(mDisplayX, win, "NVIDIA Jetson TX1 | L4T R24.1 aarch64 | Ubuntu 14.04 LTS");
	XMapWindow(mDisplayX, win);

	// cleanup
	mWindowX = win;
	mScreenX = screen;
	mVisualX = visual;
	mWidth   = screenWidth;
	mHeight  = screenHeight;
	
	XFree(fbConfig);
	return true;
}


void glDisplay::SetTitle( const char* str )
{
	XStoreName(mDisplayX, mWindowX, str);
}

// initGL
bool glDisplay::initGL()
{
	mContextGL = glXCreateContext(mDisplayX, mVisualX, 0, True);

	if( !mContextGL )
		return false;

	GL(glXMakeCurrent(mDisplayX, mWindowX, mContextGL));

	return true;
}


// MakeCurrent
void glDisplay::BeginRender()
{
	GL(glXMakeCurrent(mDisplayX, mWindowX, mContextGL));

	GL(glClearColor(0.05f, 0.05f, 0.05f, 1.0f));
	GL(glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT));

	GL(glViewport(0, 0, mWidth, mHeight));
	GL(glMatrixMode(GL_PROJECTION));
	GL(glLoadIdentity());
	GL(glOrtho(0.0f, mWidth, mHeight, 0.0f, 0.0f, 1.0f));	
}


// timeDiff
static timespec timeDiff( const timespec& start, const timespec& end)
{
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}


// Refresh
void glDisplay::EndRender()
{
	glXSwapBuffers(mDisplayX, mWindowX);

	// measure framerate
	timespec currTime;
	clock_gettime(CLOCK_REALTIME, &currTime);

	const timespec diffTime = timeDiff(mLastTime, currTime);
	const float ns = 1000000000 * diffTime.tv_sec + diffTime.tv_nsec;

	mAvgTime  = mAvgTime * 0.8f + ns * 0.2f;
	mLastTime = currTime;
}


#define MOUSE_MOVE		0
#define MOUSE_BUTTON	1
#define MOUSE_WHEEL		2
#define MOUSE_DOUBLE	3
#define KEY_STATE		4
#define KEY_CHAR		5


// OnEvent
void glDisplay::onEvent( uint msg, int a, int b )
{
	switch(msg)
	{
		case MOUSE_MOVE:
		{
			//mMousePos.Set(a,b);
			break;
		}
		case MOUSE_BUTTON:
		{
			/*if( mMouseButton[a] != (bool)b )
			{
				mMouseButton[a] = b;

				if( b )
					mMouseDownEvent = true;

				// ignore right-mouse up events
				if( !(a == 1 && !b) )
					mMouseEvent = true;
			}*/

			break;
		}
		case MOUSE_DOUBLE:
		{
			/*mMouseDblClick = b;

			if( b )
			{
				mMouseEvent = true;
				mMouseDownEvent = true;
			}*/

			break;
		}
		case MOUSE_WHEEL:
		{
			//mMouseWheel = a;
			break;
		}
		case KEY_STATE:
		{
			//mKeys[a] = b;
			break;
		}
		case KEY_CHAR:
		{
			//mKeyText = a;
			break;
		}
	}

	//if( msg == MOUSE_MOVE || msg == MOUSE_BUTTON || msg == MOUSE_DOUBLE || msg == MOUSE_WHEEL )
	//	mMouseEventLast = time();
}


// UserEvents()
void glDisplay::UserEvents()
{
	// reset input states
	/*mMouseEvent     = false;
	mMouseDownEvent = false;
	mMouseDblClick  = false;
	mMouseWheel     = 0;
	mKeyText		= 0;*/


	XEvent evt;

	while( XEventsQueued(mDisplayX, QueuedAlready) > 0 )
	{
		XNextEvent(mDisplayX, &evt);

		switch( evt.type )
		{
			case KeyPress:	     onEvent(KEY_STATE, evt.xkey.keycode, 1);		break;
			case KeyRelease:     onEvent(KEY_STATE, evt.xkey.keycode, 0);		break;
			case ButtonPress:	 onEvent(MOUSE_BUTTON, evt.xbutton.button, 1); 	break;
			case ButtonRelease:  onEvent(MOUSE_BUTTON, evt.xbutton.button, 0);	break;
			case MotionNotify:
			{
				XWindowAttributes attr;
				XGetWindowAttributes(mDisplayX, evt.xmotion.root, &attr);
				onEvent(MOUSE_MOVE, evt.xmotion.x_root + attr.x, evt.xmotion.y_root + attr.y);
				break;
			}
		}
	}
}

