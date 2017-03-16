/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __OPENGL_UTILITY_H
#define __OPENGL_UTILITY_H


#include <GL/glew.h>
#include <GL/glx.h>

#include <stdio.h>


/**
 * LOG_GL printf prefix.
 * @ingroup renderGL
 */
#define LOG_GL   			"[openGL] "



#define GL(x)				{ x; glCheckError( #x, __FILE__, __LINE__ ); }
#define GL_VERIFY(x)		{ x; if(glCheckError( #x, __FILE__, __LINE__ )) return false; }
#define GL_VERIFYN(x)		{ x; if(glCheckError( #x, __FILE__, __LINE__ )) return NULL; }
#define GL_CHECK(msg)		{ glCheckError(msg, __FILE__, __LINE__); }


/**
 * openGL error logging macros
 * @ingroup renderGL
 */
inline bool glCheckError(const char* msg, const char* file, int line)
{
	GLenum err = glGetError();

	if( err == GL_NO_ERROR )
		return false;

	const char* e = NULL;

	switch(err)
	{
		  case GL_INVALID_ENUM:          e = "invalid enum";      break;
		  case GL_INVALID_VALUE:         e = "invalid value";     break;
		  case GL_INVALID_OPERATION:     e = "invalid operation"; break;
		  case GL_STACK_OVERFLOW:        e = "stack overflow";    break;
		  case GL_STACK_UNDERFLOW:       e = "stack underflow";   break;
		  case GL_OUT_OF_MEMORY:         e = "out of memory";     break;
		#ifdef GL_TABLE_TOO_LARGE_EXT
		  case GL_TABLE_TOO_LARGE_EXT:   e = "table too large";   break;
		#endif
		#ifdef GL_TEXTURE_TOO_LARGE_EXT
		  case GL_TEXTURE_TOO_LARGE_EXT: e = "texture too large"; break;
		#endif
		  default:						 e = "unknown error";
	}

	printf(LOG_GL "Error %i - '%s'\n", (uint)err, e);
	printf(LOG_GL "   %s::%i\n", file, line );
	printf(LOG_GL "   %s\n", msg );
	
	return true;
}


/**
 * openGL error check + logging
 * @ingroup renderGL
 */
inline bool glCheckError(const char* msg)
{
	GLenum err = glGetError();

	if( err == GL_NO_ERROR )
		return false;

	const char* e = NULL;

	switch(err)
	{
		  case GL_INVALID_ENUM:          e = "invalid enum";      break;
		  case GL_INVALID_VALUE:         e = "invalid value";     break;
		  case GL_INVALID_OPERATION:     e = "invalid operation"; break;
		  case GL_STACK_OVERFLOW:        e = "stack overflow";    break;
		  case GL_STACK_UNDERFLOW:       e = "stack underflow";   break;
		  case GL_OUT_OF_MEMORY:         e = "out of memory";     break;
		#ifdef GL_TABLE_TOO_LARGE_EXT
		  case GL_TABLE_TOO_LARGE_EXT:   e = "table too large";   break;
		#endif
		#ifdef GL_TEXTURE_TOO_LARGE_EXT
		  case GL_TEXTURE_TOO_LARGE_EXT: e = "texture too large"; break;
		#endif
		  default:						 e = "unknown error";
	}

	printf(LOG_GL "%s    (error %i - %s)\n", msg, (uint)err, e);
	return true;
}



#define GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX 0x9048
#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049


/**
 * glPrintFreeMem
 * @ingroup renderGL
 */
inline void glPrintFreeMem()
{
	GLint total_mem_kb = 0;
	GLint cur_avail_mem_kb = 0;

	glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);
	glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX,&cur_avail_mem_kb);

	printf("[openGL]   GPU memory free    %i / %i kb\n", cur_avail_mem_kb, total_mem_kb);
}



#endif

