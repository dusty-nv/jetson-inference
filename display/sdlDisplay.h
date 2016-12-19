/*
 * inference-101
 * 
 * sud apt-get install libsdl2-dev libsdl2-ttf-dev
 * 
 * 
 */
 
#ifndef __SDL_VIEWPORT_H__
#define __SDL_VIEWPORT_H__

#include "glUtility.h"
#include "glTexture.h"

#include <time.h>
#include <SDL2/SDL_opengl.h>

typedef void (*keyboard_handler)(char);

/**
 * OpenGL display window / video viewer
 */
class sdlDisplay
{
friend class glTexture;
public:
	/**
	 * Create a new maximized openGL display window.
	 */
	static sdlDisplay* Create();
	static sdlDisplay* Create(int width, int height);

	/**
	 * Destroy window
	 */
	~sdlDisplay();

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
//	void onEvent( uint msg, int a, int b );

	/**
	 * Set the window title string.
	 */
	void SetTitle( const char* str );

	/**
	 * Proceess the quit.
	 */
	bool Quit( void ) { return mQuit; }

	/**
	 * Get the average frame time (in milliseconds).
	 */
	inline float GetFPS() { return 1000000000.0f / mAvgTime; }
		
    //The window we'll be rendering to
    SDL_Renderer* mRenderer;
    
    void RegisterKeyCallback(keyboard_handler callback) { handleCallback = callback; } 
protected:
	sdlDisplay();
		
	bool initWindow();
	bool initWindow(int width, int height);
	bool initGL();
    void handleKeys( unsigned char key, int x, int y );
    keyboard_handler handleCallback = 0;

	static const int screenIdx = 0;
		
    //The window we'll be rendering to
    SDL_Window* mWindow;

    //OpenGL context
    SDL_GLContext mContext;

    //Render flag
    bool mRenderQuad = true;

    //Event handler
    SDL_Event mEvent;

    //Main loop flag
    bool mQuit;

    //Fullscreen flag
    bool mFullscreen;
		
	uint32_t mWidth;
	uint32_t mHeight;

	timespec mLastTime;
	float    mAvgTime;
};

#endif

