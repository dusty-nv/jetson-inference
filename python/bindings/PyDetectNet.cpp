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
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyObject_HEAD
	detectNet::Detection det;
} PyDetection_Object;


#define DOC_DETECTION "Object Detection Result\n\n" \
				  "----------------------------------------------------------------------\n" \
				  "Data descriptors defined here:\n\n" \
				  "Area\n" \
				  "    Area of bounding box\n\n" \
				  "Bottom\n" \
				  "    Bottom bounding box coordinate\n\n" \
				  "Center\n" \
				  "    Center (x,y) coordinate of bounding box\n\n" \
				  "ClassID\n" \
				  "    Class index of the detected object\n\n" \
				  "Confidence\n" \
				  "    Confidence value of the detected object\n\n" \
				  "Height\n" \
				  "    Height of bounding box\n\n" \
				  "Instance\n" \
				  "    Instance index of the detected object\n\n" \
				  "Left\n" \
				  "    Left bounding box coordinate\n\n" \
				  "Right\n" \
				  "    Right bounding box coordinate\n\n" \
				  "Top\n" \
				  "    Top bounding box coordinate\n\n" \
				  "Width\n" \
				  "     Width of bounding box\n\n"

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

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iffff", kwlist, &classID, &conf, &left, &top, &right, &bottom))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detection.__init()__ failed to parse args tuple");
		return -1;
	}
  
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
	char str[1024];

	sprintf(str, 
		   "<detectNet.Detection object>\n"
		   "   -- ClassID: %i\n"
		   "   -- Confidence: %g\n"
		   "   -- Left:    %g\n"
		   "   -- Top:     %g\n"
		   "   -- Right:   %g\n"
		   "   -- Bottom:  %g\n"
		   "   -- Width:   %g\n"
		   "   -- Height:  %g\n"
		   "   -- Area:    %g\n"
		   "   -- Center:  (%g, %g)",
		   self->det.ClassID, self->det.Confidence, 
		   self->det.Left, self->det.Top, self->det.Right, self->det.Bottom,
		   self->det.Width(), self->det.Height(), self->det.Area(), cx, cy);

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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detection.Contains() failed to parse args tuple");
		return NULL;
	}

	PY_RETURN_BOOL(self->det.Contains(x,y));
}


// GetInstance
static PyObject* PyDetection_GetInstance( PyDetection_Object* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->det.Instance);
}

// SetInstance
static int PyDetection_SetInstance( PyDetection_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete detectNet.Detection.Instance attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	if( arg < 0 )
		arg = 0;

	self->det.Instance = arg;
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



// GetConfidence
static PyObject* PyDetection_GetConfidence( PyDetection_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->det.Confidence);
}

// SetLeft
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

// SetTop
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
	{ "Instance", (getter)PyDetection_GetInstance, (setter)PyDetection_SetInstance, "Instance index of the detected object", NULL},
	{ "ClassID", (getter)PyDetection_GetClassID, (setter)PyDetection_SetClassID, "Class index of the detected object", NULL},
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
                      "     detectnet-console.py\n" \
				  "     detectnet-camera.py\n\n" \
				  "__init__(...)\n" \
				  "     Loads an object detection model.\n\n" \
				  "     Parameters:\n" \
				  "       network (string) -- name of a built-in network to use\n" \
				  "                           see below for available options.\n\n" \
				  "       argv (strings) -- command line arguments passed to detectNet,\n" \
				  "                         see below for available options.\n\n" \
				  "       threshold (float) -- minimum detection threshold.\n" \
				  "                            default value is 0.5\n\n" \
 				  DETECTNET_USAGE_STRING


// Init
static int PyDetectNet_Init( PyDetectNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyDetectNet_Init()\n");
	
	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "ssd-mobilenet-v2";
	float threshold       = DETECTNET_DEFAULT_THRESHOLD;

	static char* kwlist[] = {"network", "argv", "threshold", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOf", kwlist, &network, &argList, &threshold))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.__init()__ failed to parse args tuple");
		return -1;
	}
    
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
		self->net = detectNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		LogVerbose(LOG_PY_INFERENCE "detectNet loading build-in network '%s'\n", network);
		
		// parse the selected built-in network
		detectNet::NetworkType networkType = detectNet::NetworkTypeFromStr(network);
		
		if( networkType == detectNet::CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid built-in network was requested");
			LogError(LOG_PY_INFERENCE "detectNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}
		
		// load the built-in network
		self->net = detectNet::Create(networkType, threshold);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet failed to load network");
		LogError(LOG_PY_INFERENCE "detectNet failed to load network\n");
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() failed to parse args tuple");
		return NULL;
	}

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// run the object detection
	detectNet::Detection* detections = NULL;

	const int numDetections = self->net->Detect(ptr, width, height, format, &detections, detectNet::OverlayFlagsFromStr(overlay));

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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() failed to parse arguments");
		return NULL;
	}

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
		if( !self->net->Overlay(input_ptr, output_ptr, width, height, format, 
						    detections_ptr.data(), detections_ptr.size(), 
						    detectNet::OverlayFlagsFromStr(overlay)) ) 
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Overlay() encountered an error");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


#define DOC_GET_THRESHOLD  "Return the minimum detection threshold.\n\n" \
				 	  "Parameters:  (none)\n\n" \
					  "Returns:\n" \
					  "  (float) -- the threshold for detection"

// GetThreshold
static PyObject* PyDetectNet_GetThreshold( PyDetectNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->net->GetThreshold());
}


#define DOC_SET_THRESHOLD  "Return the minimum detection threshold.\n\n" \
				 	  "Parameters:\n" \
					  "  (float) -- detection threshold\n\n" \
					  "Returns:  (none)"

// SetThreshold
PyObject* PyDetectNet_SetThreshold( PyDetectNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid object instance");
		return NULL;
	}
	
	float threshold = 0.0f;

	if( !PyArg_ParseTuple(args, "f", &threshold) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetThreshold() failed to parse arguments");
		return NULL;
	}
		
	self->net->SetThreshold(threshold);
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.GetClassDesc() failed to parse arguments");
		return NULL;
	}
		
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.GetClassSynset() failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassSynset(classIdx));
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetOverlayAlpha() failed to parse arguments");
		return NULL;
	}
		
	if( alpha < 0.0f || alpha > 255.0f )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetOverlayAlpha() -- provided alpha value is out-of-range");
		return NULL;
	}

	self->net->SetOverlayAlpha(alpha);

	Py_RETURN_NONE;
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
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.SetLineWidth() failed to parse arguments");
		return NULL;
	}
		
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
	{ "GetThreshold", (PyCFunction)PyDetectNet_GetThreshold, METH_NOARGS, DOC_GET_THRESHOLD},
	{ "SetThreshold", (PyCFunction)PyDetectNet_SetThreshold, METH_VARARGS, DOC_SET_THRESHOLD},     
	{ "GetNumClasses", (PyCFunction)PyDetectNet_GetNumClasses, METH_NOARGS, DOC_GET_NUM_CLASSES},
	{ "GetClassDesc", (PyCFunction)PyDetectNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetClassSynset", (PyCFunction)PyDetectNet_GetClassSynset, METH_VARARGS, DOC_GET_CLASS_SYNSET},
	{ "SetOverlayAlpha", (PyCFunction)PyDetectNet_SetOverlayAlpha, METH_VARARGS|METH_KEYWORDS, DOC_SET_OVERLAY_ALPHA},
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


