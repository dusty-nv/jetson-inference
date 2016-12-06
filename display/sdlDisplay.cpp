/*
 * inference-101
 * 
 * sud apt-get install libsdl2-dev libsdl2-ttf-dev 
 * 
 * 
 */
#include "debug.h"
#include "sdlDisplay.h"
#include <stdio.h>

void sdlDisplay::handleKeys( unsigned char key, int x, int y )
{
    switch (key)
    {
    case 'q':
    case 'Q':
        {
            mQuit = true;
            SDL_SetWindowFullscreen(mWindow, 0); // Switch out of fullscreeen
        }
        break;
    case 'f':
    case 'F':
        {
            Uint32 flag = SDL_WINDOW_FULLSCREEN;
            mFullscreen = SDL_GetWindowFlags(mWindow) & flag;
            SDL_SetWindowFullscreen(mWindow, mFullscreen ? 0 : SDL_WINDOW_FULLSCREEN);
        }
        break;
    default:
        if (handleCallback != NULL)
        {
            handleCallback(key);
        }
    }
}

// Constructor
sdlDisplay::sdlDisplay()
{
	mWindow    = NULL;
    mQuit = false;
    mFullscreen = false;
	mWidth     = 0;
	mHeight    = 0;
	mAvgTime   = 1.0f;
	handleCallback = NULL;

	clock_gettime(CLOCK_REALTIME, &mLastTime);
}


// Destructor
sdlDisplay::~sdlDisplay()
{
    //Disable text input
    SDL_StopTextInput();

    //Free resources and close SDL

    //Destroy window    
    SDL_DestroyWindow( mWindow );
    mWindow = NULL;

    //Quit SDL subsystems
    SDL_Quit();
}


// Create
sdlDisplay* sdlDisplay::Create()
{
	sdlDisplay* vp = new sdlDisplay();
	
	if( !vp )
		return NULL;
	if( !vp->initWindow() )
	{
		printf("[OpenSDL]  failed to create SDL Window.\n");
		delete vp;
		return NULL;
	}
	
	if( !vp->initGL() )
	{
		printf("[OpenSDL]  failed to initialize Open GL.\n");
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
	
    //Enable text input
    SDL_StartTextInput();

	debug_print("[OpenSDL]  sdlDisplay display window initialized\n");
	return vp;
}

sdlDisplay* sdlDisplay::Create(int width, int height)
{
	sdlDisplay* vp = new sdlDisplay();
	
	if( !vp )
		return NULL;
		
	if( !vp->initWindow(width, height) )
	{
		printf("[OpenSDL]  failed to create SDL Window.\n");
		delete vp;
		return NULL;
	}
	
	if( !vp->initGL() )
	{
		printf("[OpenSDL]  failed to initialize Open GL.\n");
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

    //Enable text input
    SDL_StartTextInput();

	debug_print("[OpenSDL]  sdlDisplay display window initialized\n");
	return vp;
}


// initWindow
bool sdlDisplay::initWindow()
{
    initWindow(640, 480);
}

bool sdlDisplay::initWindow(int width, int height)
{
    //Initialization flag
    bool success = true;

    mWidth     = width;
    mHeight    = height;

    //Initialize SDL
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
    {
        printf( "SDL could not initialize! SDL Error: %s\n", SDL_GetError() );
        success = false;
    }
    else
    {
        //Use OpenSDL 2.1
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MAJOR_VERSION, 2 );
        SDL_GL_SetAttribute( SDL_GL_CONTEXT_MINOR_VERSION, 1 );

        //Create window
//        mWindow = SDL_CreateWindow( "SDL Display", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN );
        SDL_CreateWindowAndRenderer( width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN, &mWindow, &mRenderer );

        if( mWindow == NULL )
        {
            printf( "Window could not be created! SDL Error: %s\n", SDL_GetError() );
            success = false;
        }
        else
        {
            //Create context
            mContext = SDL_GL_CreateContext( mWindow );
            if( mContext == NULL )
            {
                printf( "OpenSDL context could not be created! SDL Error: %s\n", SDL_GetError() );
                success = false;
            }
            else
            {
                //Use Vsync
                if( SDL_GL_SetSwapInterval( 1 ) < 0 )
                {
                    printf( "Warning: Unable to set VSync! SDL Error: %s\n", SDL_GetError() );
                }

                //Initialize OpenSDL
                if( !initGL() )
                {
                    printf( "Unable to initialize OpenSDL!\n" );
                    success = false;
                }
            }
        }
    }

    return true;
}

void sdlDisplay::SetTitle( const char* str )
{
    SDL_SetWindowTitle(mWindow, str); 
}

// initGL
bool sdlDisplay::initGL()
{
    bool success = true;
    GLenum error = GL_NO_ERROR;

    //Initialize Projection Matrix
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    
    //Check for error
    error = glGetError();
    if( error != GL_NO_ERROR )
    {
        success = false;
    }

    //Initialize Modelview Matrix
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    //Check for error
    error = glGetError();
    if( error != GL_NO_ERROR )
    {
        success = false;
    }
    
    //Initialize clear color
    glClearColor( 0.f, 0.f, 0.f, 1.f );
    
    //Check for error
    error = glGetError();
    if( error != GL_NO_ERROR )
    {
        success = false;
    }
    
    return success;
}

// MakeCurrent
void sdlDisplay::BeginRender()
{
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
void sdlDisplay::EndRender()
{
    //Update screen
    SDL_GL_SwapWindow( mWindow );

	// measure framerate
	timespec currTime;
	clock_gettime(CLOCK_REALTIME, &currTime);

	const timespec diffTime = timeDiff(mLastTime, currTime);
	const float ns = 1000000000 * diffTime.tv_sec + diffTime.tv_nsec;

	mAvgTime  = mAvgTime * 0.8f + ns * 0.2f;
	mLastTime = currTime;
}


// UserEvents()
void sdlDisplay::UserEvents()
{
    //Handle events on queue
    while( SDL_PollEvent( &mEvent ) != 0 )
    {
        //User requests quit
        if( mEvent.type == SDL_QUIT )
        {
            mQuit = true;
        }
        //Handle keypress with current mouse position
        else if( mEvent.type == SDL_TEXTINPUT )
        {
            int x = 0, y = 0;
            SDL_GetMouseState( &x, &y );
            handleKeys( mEvent.text.text[ 0 ], x, y );
        }
    }
}

