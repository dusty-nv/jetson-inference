/*
 * http://github.com/dusty-nv/jetson-inference
 */

#ifndef __GIE_LOGGER_H
#define __GIE_LOGGER_H

#include "NvInfer.h"
#include <stdio.h>


#define LOG_GIE "[GIE]  "


/**
 * Logger for GIE info/warning/errors
 */
class Logger : public nvinfer1::ILogger			
{
	void log( Severity severity, const char* msg ) override
	{
		if( severity != Severity::kINFO )
			printf(LOG_GIE "%s\n", msg);
	}
} gLogger;



#endif
