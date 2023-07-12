/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __OBJECT_TRACKER_H__
#define __OBJECT_TRACKER_H__


#include "detectNet.h"


/**
 * Standard command-line options able to be passed to detectNet::Create()
 * @ingroup objectTracker
 */
#define OBJECT_TRACKER_USAGE_STRING  "objectTracker arguments: \n" 	\
		  "  --tracking               flag to enable default tracker (IOU)\n"									\
		  "  --tracker=TRACKER        enable tracking with 'IOU' or 'KLT'\n"									\
		  "  --tracker-min-frames=N   the number of re-identified frames for a track to be considered valid (default: 3)\n" \
		  "  --tracker-drop-frames=N  number of consecutive lost frames before a track is dropped (default: 15)\n"  \
		  "  --tracker-overlap=N      how much IOU overlap is required for a bounding box to be matched (default: 0.5)\n\n" \
	
/**
 * Object tracker logging prefix
 * @ingroup objectTracker
 */
#define LOG_TRACKER "[tracker] "

  
/**
 * Object tracker interface
 * @ingroup objectTracker
 */
class objectTracker
{
public:
	/**
	 * Tracker type enum.
	 */
	enum Type
	{
		NONE,	/**< Tracking disabled */
		IOU,		/**< Intersection-Over-Union (IOU) tracker */
		KLT		/**< KLT tracker (only available with VPI) */
	};
	
	/**
	 * Create a new object tracker.
	 */
	static objectTracker* Create( Type type );
	
	/**
	 * Create a new object tracker by parsing the command line.
	 */
	static objectTracker* Create( int argc, char** argv );
	
	/**
	 * Load a new network instance by parsing the command line.
	 */
	static objectTracker* Create( const commandLine& cmdLine );
	
	/**
	 * Destructor
	 */
	virtual ~objectTracker();

	/**
	 * Process
	 */
	template<typename T> int Process( T* image, uint32_t width, uint32_t height, detectNet::Detection* detections, int numDetections )			{ return Process((void*)image, width, height, imageFormatFromType<T>(), detections, numDetections); }
	
	/**
	 * Process
	 */
	virtual int Process( void* image, uint32_t width, uint32_t height, imageFormat format, detectNet::Detection* detections, int numDetections ) = 0;

	/**
	 * IsEnabled
	 */
	inline bool IsEnabled() const					{ return mEnabled; }
	
	/**
	 * SetEnabled
	 */
	inline virtual void SetEnabled( bool enabled )	{ mEnabled = enabled; }
	
	/**
	 * GetType
	 */
	virtual Type GetType() const = 0;
	
	/**
	 * IsType
	 */
	inline bool IsType( Type type ) const			{ return GetType() == type; }
	
	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 				{ return OBJECT_TRACKER_USAGE_STRING; }

	/**
	 * Convert a Type enum to string.
	 */
	static const char* TypeToStr( Type type );
	
	/**
	 * Parse a Type enum from a string.
	 */
	static Type TypeFromStr( const char* str );
	
protected:
	objectTracker();
	
	bool mEnabled;
};

#endif
