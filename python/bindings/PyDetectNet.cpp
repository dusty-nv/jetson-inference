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

#include "PyTensorNet.h"
#include "PyDetectNet.h"

#include "detectNet.h"
#include "objectTrackerIOU.h"

#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyObject_HEAD
	detectNet::Detection det;
} PyDetection_Object;


#define DOC_DETECTION "Object Detection Result\n\n" \
				  "----------------------------------------------------------------------\n" \
				  "Data descriptors defined here:\n\n" \
				  "Confidence\n" \
				  "    Confidence value of the detected object\n\n" \
				  "ClassID\n" \
				  "    Class index of the detected object\n\n" \
				  "TrackID\n" \
				  "    Unique tracking ID (or -1 if untracked)\n\n" \
				  "TrackStatus\n" \
				  "    -1 for dropped, 0 for initializing, 1 for active/valid\n\n" \
				  "TrackFrames\n" \
				  "    The number of frames the object has been re-identified for\n\n" \
				  "TrackLost\n" \
				  "    The number of consecutive frames tracking has been lost for\n\n" \
				  "Width\n" \
				  "     Width of bounding box\n\n" \
				  "Height\n" \
				  "    Height of bounding box\n\n" \
				  "Left\n" \
				  "    Left bounding box coordinate\n\n" \
				  "Right\n" \
				  "    Right bounding box coordinate\n\n" \
				  "Top\n" \
				  "    Top bounding box coordinate\n\n" \
				  "Bottom\n" \
				  "    Bottom bounding box coordinate\n\n" \
				  "ROI\n" \
				  "    (Left, Top, Right, Bottom) tuple\n\n" \
				  "Center\n" \
				  "    Center (x,y) coordinate tuple\n\n" \
				  "Area\n" \
				  "    Area of bounding box\n\n"

// New
static PyObject* PyDetection_New( PyTypeObject* type, PyObject* args, PyObject* kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyDetection_New()\n");
	
	// allocate a new container
	PyDetection_Object* self = (PyDetection_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "detectNet.Detection tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->det.Reset();
	return (PyObject*)self;
}

// Init
static int PyDetection_Init( PyDetection_Object* self, PyObject* args, PyObject* kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyDetection_Init()\n");
	
	// parse arguments
	int classID = 0;
	
	float conf   = 0.0f;
	float left   = 0.0f;
	float top    = 0.0f;
	float right  = 0.0f;
	float bottom = 0.0f;

	static char* kwlist[] = {"classID", "confidence", "left", "top", "right", "bottom", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|ifffff", kwlist, &classID, &conf, &left, &top, &right, &bottom))
		return -1;
  
	if( classID < 0 )	
		classID = 0;

	// populate struct
	self->det.ClassID 	 = classID;
	self->det.Confidence = conf;
	self->det.Left 	 = left;
	self->det.Top 		 = top;
	self->det.Right 	 = right;
	self->det.Bottom 	 = bottom;

	return 0;
}

// Deallocate
static void PyDetection_Dealloc( PyDetection_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyDetection_Dealloc()\n");

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// ToString
static PyObject* PyDetection_ToString( PyDetection_Object* self )
{
	// get center coord
	float cx = 0.0f;
	float cy = 0.0f;

	self->det.Center(&cx, &cy);

	// format string
	char str[4096];

	if( self->det.TrackID >= 0 )
	{
		sprintf(str, 
			   "<detectNet.Detection object>\n"
			   "   -- Confidence:  %g\n"
			   "   -- ClassID:     %i\n"
			   "   -- TrackID:     %i\n"
			   "   -- TrackStatus: %i\n"
			   "   -- TrackFrames: %i\n"
			   "   -- TrackLost:   %i\n"
			   "   -- Left:    %g\n"
			   "   -- Top:     %g\n"
			   "   -- Right:   %g\n"
			   "   -- Bottom:  %g\n"
			   "   -- Width:   %g\n"
			   "   -- Height:  %g\n"
			   "   -- Area:    %g\n"
			   "   -- Center:  (%g, %g)",
			   self->det.Confidence, self->det.ClassID,  
			   self->det.TrackID, self->det.TrackStatus, self->det.TrackFrames, self->det.TrackLost,
			   self->det.Left, self->det.Top, self->det.Right, self->det.Bottom,
			   self->det.Width(), self->det.Height(), self->det.Area(), cx, cy);
	}
	else
	{
		sprintf(str, 
			   "<detectNet.Detection object>\n"
			   "   -- Confidence: %g\n"
			   "   -- ClassID: %i\n"
			   "   -- Left:    %g\n"
			   "   -- Top:     %g\n"
			   "   -- Right:   %g\n"
			   "   -- Bottom:  %g\n"
			   "   -- Width:   %g\n"
			   "   -- Height:  %g\n"
			   "   -- Area:    %g\n"
			   "   -- Center:  (%g, %g)",
			   self->det.Confidence, self->det.ClassID,  
			   self->det.Left, self->det.Top, self->det.Right, self->det.Bottom,
			   self->det.Width(), self->det.Height(), self->det.Area(), cx, cy);
	}
	
	return PYSTRING_FROM_STRING(str);
}

// Contains
static PyObject* PyDetection_Contains( PyDetection_Object* self, PyObject *args, PyObject *kwds )
{
	if( !self )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detection invalid object instance");
		return NULL;
	}

	// parse arguments
	float x = 0.0f;
	float y = 0.0f;

	static char* kwlist[] = {"x", "y", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "ff", kwlist, &x, &y))
		return NULL;

	PY_RETURN_BOOL(self->det.Contains(x,y));
}

// GetConfidence
static PyObject* PyDetection_GetConfidence( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Confidence);
}

// SetConfidence
static int PyDetection_SetConfidence( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Confidence attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.Confidence = arg;
	return 0;
}

// GetClassID
static PyObject* PyDetection_GetClassID( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->det.ClassID);
}

// SetClassID
static int PyDetection_SetClassID( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.ClassID attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	if( arg < 0 )
		arg = 0;

	self->det.ClassID = arg;
	return 0;
}

// GetTrackID
static PyObject* PyDetection_GetTrackID( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_LONG(self->det.TrackID);
}

// SetTrackID
static int PyDetection_SetTrackID( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.TrackID attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.TrackID = arg;
	return 0;
}

// GetTrackStatus
static PyObject* PyDetection_GetTrackStatus( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_LONG(self->det.TrackStatus);
}

// SetTrackStatus
static int PyDetection_SetTrackStatus( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.TrackStatus attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.TrackStatus = arg;
	return 0;
}

// GetTrackFrames
static PyObject* PyDetection_GetTrackFrames( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_LONG(self->det.TrackFrames);
}

// SetTrackFrames
static int PyDetection_SetTrackFrames( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.TrackFrames attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.TrackFrames = arg;
	return 0;
}

// GetTrackLost
static PyObject* PyDetection_GetTrackLost( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_LONG(self->det.TrackLost);
}

// SetTrackLost
static int PyDetection_SetTrackLost( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.TrackLost attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.TrackLost = arg;
	return 0;
}

// GetLeft
static PyObject* PyDetection_GetLeft( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Left);
}

// SetLeft
static int PyDetection_SetLeft( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Left attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.Left = arg;
	return 0;
}

// GetRight
static PyObject* PyDetection_GetRight( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Right);
}

// SetRight
static int PyDetection_SetRight( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Right attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.Right = arg;
	return 0;
}

// GetTop
static PyObject* PyDetection_GetTop( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Top);
}

// SetTop
static int PyDetection_SetTop( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Top attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.Top = arg;
	return 0;
}

// GetBottom
static PyObject* PyDetection_GetBottom( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Bottom);
}

// SetBottom
static int PyDetection_SetBottom( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Bottom attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->det.Bottom = arg;
	return 0;
}

// GetWidth
static PyObject* PyDetection_GetWidth( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Width());
}

// GetHeight
static PyObject* PyDetection_GetHeight( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Height());
}

// GetArea
static PyObject* PyDetection_GetArea( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Area());
}

// GetCenter
static PyObject* PyDetection_GetCenter( PyDetection_Object* self, void* closure )
{
	float x = 0.0f;
	float y = 0.0f;

	self->det.Center(&x, &y);

	// create tuple objects
	PyObject* centerX = PyFloat_FromDouble(x);
	PyObject* centerY = PyFloat_FromDouble(y);

	PyObject* tuple = PyTuple_Pack(2, centerX, centerY);

	Py_DECREF(centerX);
	Py_DECREF(centerY);

	return tuple;
}

// GetROI
static PyObject* PyDetection_GetROI( PyDetection_Object* self, void* closure )
{
	PyObject* left   = PyFloat_FromDouble(self->det.Left);
	PyObject* top    = PyFloat_FromDouble(self->det.Top);
	PyObject* right  = PyFloat_FromDouble(self->det.Right);
	PyObject* bottom = PyFloat_FromDouble(self->det.Bottom);

	PyObject* tuple = PyTuple_Pack(4, left, top, right, bottom);

	Py_DECREF(left);
	Py_DECREF(top);
	Py_DECREF(right);
	Py_DECREF(bottom);

	return tuple;
}

static PyGetSetDef pyDetection_GetSet[] = 
{
	{ "ClassID", (getter)PyDetection_GetClassID, (setter)PyDetection_SetClassID, "Class index of the detected object", NULL},
	{ "TrackID", (getter)PyDetection_GetTrackID, (setter)PyDetection_SetTrackID, "Unique tracking ID (-1 if untracked)", NULL},
	{ "TrackStatus", (getter)PyDetection_GetTrackStatus, (setter)PyDetection_SetTrackStatus, "-1 for dropped, 0 for initializing, 1 for active/valid", NULL},
	{ "TrackFrames", (getter)PyDetection_GetTrackFrames, (setter)PyDetection_SetTrackFrames, "The number of frames the object has been re-identified for", NULL},
	{ "TrackLost", (getter)PyDetection_GetTrackLost, (setter)PyDetection_SetTrackLost, "The number of consecutive frames tracking has been lost for", NULL},
	{ "Instance", (getter)PyDetection_GetTrackID, (setter)PyDetection_SetTrackID, "Unique tracking ID (-1 if untracked)", NULL}, // legacy
	{ "Confidence", (getter)PyDetection_GetConfidence, (setter)PyDetection_SetConfidence, "Confidence value of the detected object", NULL},
	{ "Left", (getter)PyDetection_GetLeft, (setter)PyDetection_SetLeft, "Left bounding box coordinate", NULL},
	{ "Right", (getter)PyDetection_GetRight, (setter)PyDetection_SetRight, "Right bounding box coordinate", NULL},
	{ "Top", (getter)PyDetection_GetTop, (setter)PyDetection_SetTop, "Top bounding box coordinate", NULL},
	{ "Bottom", (getter)PyDetection_GetBottom, (setter)PyDetection_SetBottom, "Bottom bounding box coordinate", NULL},	
	{ "Width", (getter)PyDetection_GetWidth, NULL, "Width of bounding box", NULL},
	{ "Height", (getter)PyDetection_GetHeight, NULL, "Height of bounding box", NULL},
	{ "Area", (getter)PyDetection_GetArea, NULL, "Area of bounding box", NULL},
	{ "Center", (getter)PyDetection_GetCenter, NULL, "Center (x,y) coordinate of bounding box", NULL},
	{ "ROI", (getter)PyDetection_GetROI, NULL, "Tuple containing the ROI as (Left, Top, Right, Bottom)", NULL},
	{ NULL } /* Sentinel */
};

static PyMethodDef pyDetection_Methods[] = 
{
	{ "Contains", (PyCFunction)PyDetection_Contains, METH_VARARGS|METH_KEYWORDS, "Return true if the given coordinate lies inside of the bounding box"},
	{ NULL }  /* Sentinel */
};

static PyTypeObject pyDetection_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};



//-----------------------------------------------------------------------------------------
typedef struct {
	PyTensorNet_Object base;
	detectNet* net;
} PyDetectNet_Object;


#define DOC_DETECTNET "Object Detection DNN - locates objects in an image\n\n" \
				  "Examples (jetson-inference/python/examples)\n" \
                      "     detectnet.py\n\n" \
				  "__init__(...)\n" \
				  "     Loads an object detection model.\n\n" \
				  "     Parameters:\n" \
				  "       network (string) -- name of a built-in network to use\n" \
				  "                           see below for available options.\n\n" \
				  "       argv (strings) -- command line arguments passed to detectNet,\n" \
				  "                         see below for available options.\n\n" \
				  "       threshold (float) -- minimum detection threshold.\n" \
				  "                            default value is 0.5\n\n" \
				  "     Extended parameters for loading custom models:\n" \
				  "       model (string) -- path to self-trained ONNX model to load.\n\n" \
				  "       labels (string) -- path to labels.txt file (optional)\n\n" \
				  "       colors (string) -- path to colors.txt file (optional)\n\n" \
				  "       input_blob (string) -- name of the input layer of the model.\n\n" \
				  "       output_cvg (string) -- name of the output coverage/confidence layer.\n\n" \
				  "       output_bbox (string) -- name of the output bounding boxes layer.\n\n" \
 				  DETECTNET_USAGE_STRING

// Init
static int PyDetectNet_Init( PyDetectNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyDetectNet_Init()\n");
	
	// parse arguments
	PyObject* argList = NULL;

	const char* network     = "ssd-mobilenet-v2";
	const char* model       = NULL;
	const char* labels      = NULL;
	const char* colors      = NULL;
	const char* input_blob  = DETECTNET_DEFAULT_INPUT;
	const char* output_cvg  = DETECTNET_DEFAULT_COVERAGE;
	const char* output_bbox = DETECTNET_DEFAULT_BBOX;
	
	float threshold = DETECTNET_DEFAULT_CONFIDENCE_THRESHOLD;
	
	static char* kwlist[] = {"network", "argv", "threshold", "model", "labels", "colors", "input_blob", "output_cvg", "output_bbox", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOfssssss", kwlist, &network, &argList, &threshold, &model, &labels, &colors, &input_blob, &output_cvg, &output_bbox))
		return -1;
    
	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		LogDebug(LOG_PY_INFERENCE "detectNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "detectNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "detectNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		Py_BEGIN_ALLOW_THREADS
		self->net = detectNet::Create(argc, argv);
		Py_END_ALLOW_THREADS

		// free the arguments array
		free(argv);
	}
	else
	{
		LogVerbose(LOG_PY_INFERENCE "detectNet loading custom model '%s'\n", model);
		
		// load the network using custom model parameters
		Py_BEGIN_ALLOW_THREADS
		self->net = detectNet::Create(NULL, model != NULL ? model : network, 0.0f, labels, colors, threshold, input_blob, output_cvg, output_bbox);
		Py_END_ALLOW_THREADS
	}	
	
	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet failed to load network");
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyDetectNet_Dealloc( PyDetectNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyDetectNet_Dealloc()\n");

	// delete the network
	SAFE_DELETE(self->net);
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_DETECT   "Detect objects in an RGBA image and return a list of detections.\n\n" \
				 "Parameters:\n" \
				 "  image   (capsule) -- CUDA memory capsule\n" \
				 "  width   (int)  -- width of the image (in pixels)\n" \
				 "  height  (int)  -- height of the image (in pixels)\n" \
				 "  overlay (str)  -- combination of box,lines,labels,conf,none flags (default is 'box,labels,conf')\n\n" \
				 "Returns:\n" \
				 "  [Detections] -- list containing the detected objects (see detectNet.Detection)"

// Detect
static PyObject* PyDetectNet_Detect( PyDetectNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* overlay    = "box,labels,conf";
	const char* format_str = "rgba32f";
	static char* kwlist[]  = {"image", "width", "height", "overlay", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iiss", kwlist, &capsule, &width, &height, &overlay, &format_str))
		return NULL;

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// run the object detection
	detectNet::Detection* detections = NULL;
	int numDetections = 0;
	
	Py_BEGIN_ALLOW_THREADS
	numDetections = self->net->Detect(ptr, width, height, format, &detections, detectNet::OverlayFlagsFromStr(overlay));
	Py_END_ALLOW_THREADS
	
	if( numDetections < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() encountered an error classifying the image");
		return NULL;
	}

	// create output objects
	PyObject* list = PyList_New(numDetections);

	for( uint32_t n=0; n < numDetections; n++ )
	{
		PyDetection_Object* pyDetection = PyObject_New(PyDetection_Object, &pyDetection_Type);

		if( !pyDetection )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() failed to create a new detectNet.Detection object");
			return NULL;
		}

		pyDetection->det = detections[n];
		PyList_SET_ITEM(list, n, (PyObject*)pyDetection);
	}

	return list;
}

#define DOC_OVERLAY "Overlay a list of detections in an RGBA image.\n\n" \
				 "Parameters:\n" \
				 "  image   (capsule) -- CUDA memory capsule\n" \
				 "  [Detections]   -- list containing the detected objects (see detectNet.Detection)" \
				 "  width   (int)  -- width of the image (in pixels)\n" \
				 "  height  (int)  -- height of the image (in pixels)\n" \
				 "  overlay (str)  -- combination of box,lines,labels,conf,none flags (default is 'box,labels,conf')\n\n" \
				 "Returns:\n" \
				 "  None"

// Overlay
static PyObject* PyDetectNet_Overlay( PyDetectNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* input_capsule = NULL;
	PyObject* output_capsule = NULL;
	PyObject* detections = NULL;

	int width = 0;
	int height = 0;

	const char* overlay    = "box,labels,conf";
	const char* format_str = "rgba32f";
	static char* kwlist[]  = {"image", "detections", "width", "height", "overlay", "format", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|iissO", kwlist, &input_capsule, &detections, &width, &height, &overlay, &format_str, &output_capsule))
		return NULL;

	if( !output_capsule )
		output_capsule = input_capsule;
	
	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* input_ptr = PyCUDA_GetImage(input_capsule, &width, &height, &format);
	void* output_ptr = PyCUDA_GetImage(output_capsule, &width, &height, &format);

	if( !input_ptr || !output_ptr ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from input or output image argument(s)");
		return NULL;
	}
	
	if( !PyList_Check(detections) ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "detections should be of type list");
		return NULL;
	}

	auto detections_ptr = std::vector<detectNet::Detection>();

	for( Py_ssize_t i=0; i < PyList_Size(detections); i++ ) 
	{
		PyObject* value = PyList_GetItem(detections, i);

		if( PyObject_IsInstance(value, (PyObject*)&pyDetection_Type) != 1 ) 
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "detections value should be of type jetson.inference.detectNet.Detection");
			return NULL;
		}
		
		detections_ptr.push_back(((PyDetection_Object*)value)->det);
	}

	if( detections_ptr.size() > 0 ) 
	{
		bool result = false;
		Py_BEGIN_ALLOW_THREADS
		
		result = self->net->Overlay(input_ptr, output_ptr, width, height, format, 
						    detections_ptr.data(), detections_ptr.size(), 
						    detectNet::OverlayFlagsFromStr(overlay));
				
		Py_END_ALLOW_THREADS
		
		if( !result ) 
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Overlay() encountered an error");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


#define DOC_GET_CONFIDENCE_THRESHOLD  "Return the minimum detection threshold.\n\n" \
							   "Parameters:  (none)\n\n" \
							   "Returns:\n" \
							   "  (float) -- the threshold for detection"

// GetConfidenceThreshold
static PyObject* PyDetectNet_GetConfidenceThreshold( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->net->GetConfidenceThreshold());
}


#define DOC_SET_CONFIDENCE_THRESHOLD  "Set the minimum detection threshold.\n\n" \
				 	             "Parameters:\n" \
					             "  (float) -- detection threshold\n\n" \
					             "Returns:  (none)"

// SetConfidenceThreshold
PyObject* PyDetectNet_SetConfidenceThreshold( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	float threshold = 0.0f;

	if( !PyArg_ParseTuple(args, "f", &threshold) )
		return NULL;

	self->net->SetConfidenceThreshold(threshold);
	Py_RETURN_NONE;
}


#define DOC_GET_CLUSTERING_THRESHOLD  "Return the overlapping area % threshold for clustering.\n\n" \
							   "Parameters:  (none)\n\n" \
							   "Returns:\n" \
							   "  (float) -- the overlapping area % threshold for merging bounding boxes"

// GetClusteringThreshold
static PyObject* PyDetectNet_GetClusteringThreshold( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->net->GetClusteringThreshold());
}


#define DOC_SET_CLUSTERING_THRESHOLD  "Set the overlapping area % threshold for clustering.\n\n" \
				 	             "Parameters:\n" \
					             "  (float) -- the overlapping area % threshold for merging bounding boxes\n\n" \
					             "Returns:  (none)"

// SetClusteringThreshold
PyObject* PyDetectNet_SetClusteringThreshold( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	float threshold = 0.0f;

	if( !PyArg_ParseTuple(args, "f", &threshold) )
		return NULL;

	self->net->SetClusteringThreshold(threshold);
	Py_RETURN_NONE;
}


#define DOC_GET_NUM_CLASSES "Return the number of object classes that this network model is able to detect.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (int) -- number of object classes that the model supports"

// GetNumClasses
static PyObject* PyDetectNet_GetNumClasses( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->net->GetNumClasses());
}


#define DOC_GET_CLASS_DESC "Return the class description for the given object class.\n\n" \
				 	  "Parameters:\n" \
					  "  (int) -- index of the class, between [0, GetNumClasses()]\n\n" \
					  "Returns:\n" \
					  "  (string) -- the text description of the object class"

// GetClassDesc
PyObject* PyDetectNet_GetClassDesc( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
		return NULL;
	
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassDesc(classIdx));
}


#define DOC_GET_CLASS_SYNSET "Return the synset data category string for the given class.\n" \
					    "The synset generally maps to the class training data folder.\n\n" \
				 	    "Parameters:\n" \
					    "  (int) -- index of the class, between [0, GetNumClasses()]\n\n" \
					    "Returns:\n" \
					    "  (string) -- the synset of the class, typically 9 characters long" 

// GetClassSynset
PyObject* PyDetectNet_GetClassSynset( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
		return NULL;
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassSynset(classIdx));
}


#define DOC_GET_TRACKER_TYPE "Returns the type of tracker being used as a string\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- 'IOU', 'KLT', or None if no tracking\n"

// GetTrackerType
static PyObject* PyDetectNet_GetTrackerType( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	objectTracker* tracker = self->net->GetTracker();
	
	if( !tracker )
		Py_RETURN_NONE;
	
	return Py_BuildValue("s", objectTracker::TypeToStr(tracker->GetType()));
}


#define DOC_SET_TRACKER_TYPE "Sets the type of tracker being used\n\n" \
				 	    "Parameters:\n" \
					    "  (string) -- 'IOU' or 'KLT' (other strings will disable tracking)\n" \
					    "Returns:  (none)"

// SetTrackerType
static PyObject* PyDetectNet_SetTrackerType( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	const char* typeStr = NULL;

	if( !PyArg_ParseTuple(args, "s", &typeStr) )
		return NULL;

	const objectTracker::Type type = objectTracker::TypeFromStr(typeStr);
	
	// either create a new tracker, or save the existing one if the requested type matches
	objectTracker* tracker = self->net->GetTracker();
	
	if( tracker != NULL && tracker->IsType(type) )
		Py_RETURN_NONE;
	
	if( tracker != NULL )
		delete tracker;
	
	self->net->SetTracker(objectTracker::Create(type));
	
	Py_RETURN_NONE;
}


#define DOC_IS_TRACKING_ENABLED "Returns true if tracking is enabled, otherwise false\n\n" \
					       "Parameters:  (none)\n\n" \
					       "Returns:\n" \
					       "  (bool) -- true if tracking is enabled, otherwise false\n"

// IsTrackingEnabled
static PyObject* PyDetectNet_IsTrackingEnabled( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	objectTracker* tracker = self->net->GetTracker();
	
	if( tracker != NULL && tracker->IsEnabled() )
		Py_RETURN_TRUE;
	
	Py_RETURN_FALSE;
}


#define DOC_SET_TRACKING_ENABLED "Sets if tracking is enabled or disabled.\n" \
						   "When enabling tracking, if the tracker type wasn't previously\n" \
						   "set with detectNet.SetTrackerType(), then 'IOU' will be used.\n\n" \
				 	        "Parameters:\n" \
					        "  (bool) -- true to enable tracking, false to disable it\n" \
					        "Returns:  (none)"

// SetTrackingEnabled
static PyObject* PyDetectNet_SetTrackingEnabled( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	int enabled = 0;

	if( !PyArg_ParseTuple(args, "p", &enabled) )
		return NULL;

	objectTracker* tracker = self->net->GetTracker();

	if( tracker != NULL )
	{
		tracker->SetEnabled((bool)enabled);
		Py_RETURN_NONE;
	}
	
	if( !enabled )
		Py_RETURN_NONE;
	
	self->net->SetTracker(objectTracker::Create(objectTracker::IOU));
	
	Py_RETURN_NONE;
}


#define DOC_GET_TRACKING_PARAMS "Returns a dict containing various tracking parameters.\n\n" \
				 "Parameters: (none)\n\n" \
				 "Returns: a dict containing the following keys/values (dependent on the type of tracker):\n" \
				 "  minFrames (int) -- the number of re-identified frames before before establishing a track (IOU tracker only)\n" \
				 "  dropFrames (int) -- the number of consecutive lost frames after which a track is removed (IOU tracker only)\n" \
				 "  overlapThreshold (float) -- how much IOU overlap is required for a bounding box to be matched, between [0,1] (IOU tracker only)\n"
				 
// GetTrackingParams
static PyObject* PyDetectNet_GetTrackingParams( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	PyObject* dict = PyDict_New();
	
	objectTracker* tracker = self->net->GetTracker();
	
	if( !tracker )
	{
		PYDICT_SET_UINT(dict, "minFrames", OBJECT_TRACKER_DEFAULT_MIN_FRAMES);
		PYDICT_SET_UINT(dict, "dropFrames", OBJECT_TRACKER_DEFAULT_DROP_FRAMES);
		PYDICT_SET_FLOAT(dict, "overlapThreshold", OBJECT_TRACKER_DEFAULT_OVERLAP_THRESHOLD);
	}
	else if( tracker->IsType(objectTracker::IOU) )
	{
		objectTrackerIOU* iou = (objectTrackerIOU*)tracker;
		
		PYDICT_SET_UINT(dict, "minFrames", iou->GetMinFrames());
		PYDICT_SET_UINT(dict, "dropFrames", iou->GetDropFrames());
		PYDICT_SET_FLOAT(dict, "overlapThreshold", iou->GetOverlapThreshold());
	}
	
	return dict;
}


#define DOC_SET_TRACKING_PARAMS "Sets various tracker parameters using keyword arguments.\n\n" \
				 "Parameters:\n" \
				 "  minFrames (int) -- the number of re-identified frames before before establishing a track (IOU tracker only)\n" \
				 "  dropFrames (int) -- the number of consecutive lost frames after which a track is removed (IOU tracker only)\n" \
				 "  overlapThreshold (float) -- how much IOU overlap is required for a bounding box to be matched, between [0,1] (IOU tracker only)\n\n" \
				 "Returns:\n" \
				 "  None"

// SetTrackingParams
static PyObject* PyDetectNet_SetTrackingParams( PyDetectNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	int minFrames = -1;
	int dropFrames = -1;
	float overlapThreshold = -1;

	static char* kwlist[]  = {"minFrames", "dropFrames", "overlapThreshold", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iif", kwlist, &minFrames, &dropFrames, &overlapThreshold) )
		return NULL;
	
	objectTracker* tracker = self->net->GetTracker();
	
	if( !tracker )
		Py_RETURN_NONE;
	
	if( tracker->IsType(objectTracker::IOU) )
	{
		objectTrackerIOU* iou = (objectTrackerIOU*)tracker;
		
		if( minFrames >= 0 )
			iou->SetMinFrames(minFrames);
		
		if( dropFrames >= 0 )
			iou->SetDropFrames(dropFrames);
		
		if( overlapThreshold >= 0 )
			iou->SetOverlapThreshold(overlapThreshold);
	}
	
	Py_RETURN_NONE;
}


#define DOC_GET_OVERLAY_ALPHA "Return the overlay alpha blending value for classes that don't have it explicitly set.\n\n" \
				 	     "Parameters: (none)\n\n" \
					     "Returns:\n" \
					     "  (float) -- alpha blending value between [0,255]"

// GetOverlayAlpha
PyObject* PyDetectNet_GetOverlayAlpha( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	return PyFloat_FromDouble(self->net->GetOverlayAlpha());
}


#define DOC_SET_OVERLAY_ALPHA "Set the alpha blending value used during overlay visualization for all classes\n\n" \
				 	  "Parameters:\n" \
					  "  alpha (float) -- desired alpha value, between 0.0 and 255.0\n" \
					  "Returns:  (none)"

// SetOverlayAlpha
PyObject* PyDetectNet_SetOverlayAlpha( PyDetectNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	float alpha = 0.0f;
	static char* kwlist[] = {"alpha", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f", kwlist, &alpha) )
		return NULL;

	if( alpha < 0.0f || alpha > 255.0f )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetOverlayAlpha() -- provided alpha value is out-of-range");
		return NULL;
	}

	self->net->SetOverlayAlpha(alpha);

	Py_RETURN_NONE;
}


#define DOC_GET_LINE_WIDTH "Return the line width used during overlay when 'lines' mode is used.\n\n" \
				 	  "Parameters: (none)\n\n" \
					  "Returns:\n" \
					  "  (float) -- line width in pixels"

// GetLineWidth
PyObject* PyDetectNet_GetLineWidth( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	return PyFloat_FromDouble(self->net->GetLineWidth());
}


#define DOC_SET_LINE_WIDTH "Set the line width used during overlay when 'lines' mode is used\n\n" \
				 	  "Parameters:\n" \
					  "  width (float) -- desired line width, in pixels\n" \
					  "Returns:  (none)"

// SetOverlayAlpha
PyObject* PyDetectNet_SetLineWidth( PyDetectNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	float width = 0.0f;
	static char* kwlist[] = {"width", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f", kwlist, &width) )
		return NULL;
	
	if( width <= 0.0f )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetLineWidth() -- provided value is out-of-range");
		return NULL;
	}

	self->net->SetLineWidth(width);

	Py_RETURN_NONE;
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyDetectNet_Usage( PyDetectNet_Object* self )
{
	return Py_BuildValue("s", detectNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject pyDetectNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyDetectNet_Methods[] = 
{
	{ "Detect", (PyCFunction)PyDetectNet_Detect, METH_VARARGS|METH_KEYWORDS, DOC_DETECT},
	{ "Overlay", (PyCFunction)PyDetectNet_Overlay, METH_VARARGS|METH_KEYWORDS, DOC_OVERLAY},
	{ "GetNumClasses", (PyCFunction)PyDetectNet_GetNumClasses, METH_NOARGS, DOC_GET_NUM_CLASSES},
	{ "GetClassLabel", (PyCFunction)PyDetectNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetClassDesc", (PyCFunction)PyDetectNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetClassSynset", (PyCFunction)PyDetectNet_GetClassSynset, METH_VARARGS, DOC_GET_CLASS_SYNSET},
	{ "GetThreshold", (PyCFunction)PyDetectNet_GetConfidenceThreshold, METH_NOARGS, DOC_GET_CONFIDENCE_THRESHOLD},
	{ "SetThreshold", (PyCFunction)PyDetectNet_SetConfidenceThreshold, METH_VARARGS, DOC_SET_CONFIDENCE_THRESHOLD}, 
	{ "GetConfidenceThreshold", (PyCFunction)PyDetectNet_GetConfidenceThreshold, METH_NOARGS, DOC_GET_CONFIDENCE_THRESHOLD},
	{ "SetConfidenceThreshold", (PyCFunction)PyDetectNet_SetConfidenceThreshold, METH_VARARGS, DOC_SET_CONFIDENCE_THRESHOLD}, 
	{ "GetClusteringThreshold", (PyCFunction)PyDetectNet_GetClusteringThreshold, METH_NOARGS, DOC_GET_CLUSTERING_THRESHOLD},
	{ "SetClusteringThreshold", (PyCFunction)PyDetectNet_SetClusteringThreshold, METH_VARARGS, DOC_SET_CLUSTERING_THRESHOLD},
	{ "GetTrackerType", (PyCFunction)PyDetectNet_GetTrackerType, METH_NOARGS, DOC_GET_TRACKER_TYPE},
	{ "SetTrackerType", (PyCFunction)PyDetectNet_SetTrackerType, METH_VARARGS, DOC_SET_TRACKER_TYPE},
	{ "IsTrackingEnabled", (PyCFunction)PyDetectNet_IsTrackingEnabled, METH_NOARGS, DOC_IS_TRACKING_ENABLED},
	{ "SetTrackingEnabled", (PyCFunction)PyDetectNet_SetTrackingEnabled, METH_VARARGS, DOC_SET_TRACKING_ENABLED},
	{ "GetTrackingParams", (PyCFunction)PyDetectNet_GetTrackingParams, METH_NOARGS, DOC_GET_TRACKING_PARAMS},
	{ "SetTrackingParams", (PyCFunction)PyDetectNet_SetTrackingParams, METH_VARARGS|METH_KEYWORDS, DOC_SET_TRACKING_PARAMS},
	{ "GetOverlayAlpha", (PyCFunction)PyDetectNet_GetOverlayAlpha, METH_NOARGS, DOC_GET_OVERLAY_ALPHA},
	{ "SetOverlayAlpha", (PyCFunction)PyDetectNet_SetOverlayAlpha, METH_VARARGS|METH_KEYWORDS, DOC_SET_OVERLAY_ALPHA},
	{ "GetLineWidth", (PyCFunction)PyDetectNet_GetLineWidth, METH_NOARGS, DOC_GET_LINE_WIDTH},
	{ "SetLineWidth", (PyCFunction)PyDetectNet_SetLineWidth, METH_VARARGS|METH_KEYWORDS, DOC_SET_LINE_WIDTH},
	{ "Usage", (PyCFunction)PyDetectNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},	
	{NULL}  /* Sentinel */
};

// Register type
bool PyDetectNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	/*
	 * register detectNet.Detection type
	 */
	pyDetection_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".detectNet.Detection";
	pyDetection_Type.tp_basicsize	= sizeof(PyDetection_Object);
	pyDetection_Type.tp_flags	= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyDetection_Type.tp_base		= NULL;
	pyDetection_Type.tp_methods	= pyDetection_Methods;
	pyDetection_Type.tp_getset    = pyDetection_GetSet;
	pyDetection_Type.tp_new		= PyDetection_New;
	pyDetection_Type.tp_init		= (initproc)PyDetection_Init;
	pyDetection_Type.tp_dealloc	= (destructor)PyDetection_Dealloc;
	pyDetection_Type.tp_str		= (reprfunc)PyDetection_ToString;
	pyDetection_Type.tp_repr		= (reprfunc)PyDetection_ToString;
	pyDetection_Type.tp_doc		= DOC_DETECTION;

	if( PyType_Ready(&pyDetection_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "detectNet.Detection PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyDetection_Type);


	/*
	 * register detectNet type
	 */
	pyDetectNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".detectNet";
	pyDetectNet_Type.tp_basicsize	= sizeof(PyDetectNet_Object);
	pyDetectNet_Type.tp_flags	= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyDetectNet_Type.tp_base		= PyTensorNet_Type();
	pyDetectNet_Type.tp_methods	= pyDetectNet_Methods;
	pyDetectNet_Type.tp_new		= NULL; /*PyDetectNet_New;*/
	pyDetectNet_Type.tp_init		= (initproc)PyDetectNet_Init;
	pyDetectNet_Type.tp_dealloc	= (destructor)PyDetectNet_Dealloc;
	pyDetectNet_Type.tp_doc		= DOC_DETECTNET;
	 
	// setup Detection as inner class for detectNet object
	pyDetectNet_Type.tp_dict = PyDict_New();

	if( !pyDetectNet_Type.tp_dict )
	{
		LogError(LOG_PY_INFERENCE "detectNet failed to create new PyDict object\n");
		return false;
	}

	if( PyDict_SetItemString(pyDetectNet_Type.tp_dict, "Detection", (PyObject*)&pyDetection_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "detectNet failed to register detectNet.Detection inner class\n");
		return false;
	}

	// complete registration of the detectNet type
	if( PyType_Ready(&pyDetectNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "detectNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyDetectNet_Type);

	if( PyModule_AddObject(module, "detectNet", (PyObject*)&pyDetectNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "detectNet PyModule_AddObject('detectNet') failed\n");
		return false;
	}
	
	return true;
}


