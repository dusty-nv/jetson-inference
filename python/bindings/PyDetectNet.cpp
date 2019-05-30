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

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyObject_HEAD
	detectNet::Detection det;
} PyDetection_Object;


// New
static PyObject* PyDetection_New( PyTypeObject* type, PyObject* args, PyObject* kwds )
{
	printf(LOG_PY_INFERENCE "PyDetection_New()\n");
	
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
	printf(LOG_PY_INFERENCE "PyDetection_Init()\n");
	
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
	printf(LOG_PY_INFERENCE "PyDetection_Dealloc()\n");

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

static PyGetSetDef pyDetection_GetSet[] = 
{
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


// Init
static int PyDetectNet_Init( PyDetectNet_Object* self, PyObject *args, PyObject *kwds )
{
	printf(LOG_PY_INFERENCE "PyDetectNet_Init()\n");
	
	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "multiped";
	float threshold       = 0.5f;

	static char* kwlist[] = {"network", "argv", "threshold", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOf", kwlist, &network, &argList, &threshold))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.__init()__ failed to parse args tuple");
		return -1;
	}
    
	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		printf(LOG_PY_INFERENCE "detectNet loading network using argv command line params\n");

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

			printf(LOG_PY_INFERENCE "detectNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		self->net = detectNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		printf(LOG_PY_INFERENCE "detectNet loading build-in network '%s'\n", network);
		
		// parse the selected built-in network
		detectNet::NetworkType networkType = detectNet::NetworkTypeFromStr(network);
		
		if( networkType == detectNet::CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet invalid built-in network was requested");
			printf(LOG_PY_INFERENCE "detectNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}
		
		// load the built-in network
		self->net = detectNet::Create(networkType, threshold);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet failed to load network");
		printf(LOG_PY_INFERENCE "detectNet failed to load built-in network '%s'\n", network);
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


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
	int overlay = 1;

	static char* kwlist[] = {"image", "width", "height", "overlay", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|i", kwlist, &capsule, &width, &height, &overlay))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() failed to parse args tuple");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() image dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "detectNet.Detect() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// run the object detection
	detectNet::Detection* detections = NULL;

	const int numDetections = self->net->Detect((float*)img, width, height, &detections, overlay > 0 ? true : false);

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


//-------------------------------------------------------------------------------
static PyTypeObject pyDetectNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyDetectNet_Methods[] = 
{
	{ "Detect", (PyCFunction)PyDetectNet_Detect, METH_VARARGS|METH_KEYWORDS, "Detect objects in an RGBA image and optionally overlay the detected bounding boxes over the image"},
	{ "GetThreshold", (PyCFunction)PyDetectNet_GetThreshold, METH_NOARGS, "Return the minimum threshold for detection"},
	{ "SetThreshold", (PyCFunction)PyDetectNet_SetThreshold, METH_VARARGS, "Set the minimum threshold for detection"},     
	{ "GetNumClasses", (PyCFunction)PyDetectNet_GetNumClasses, METH_NOARGS, "Return the number of object classes that this network model is able to classify"},
	{ "GetClassDesc", (PyCFunction)PyDetectNet_GetClassDesc, METH_VARARGS, "Return the class description for the given class index"},
	{ "GetClassSynset", (PyCFunction)PyDetectNet_GetClassSynset, METH_VARARGS, "Return the class synset dataset category for the given class index"},
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
	pyDetection_Type.tp_doc		= "Object Detection Result";

	if( PyType_Ready(&pyDetection_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "detectNet.Detection PyType_Ready() failed\n");
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
	pyDetectNet_Type.tp_dealloc	= NULL; /*(destructor)PyDetectNet_Dealloc;*/
	pyDetectNet_Type.tp_doc		= "Object Detection DNN";
	 
	// setup Detection as inner class for detectNet object
	pyDetectNet_Type.tp_dict = PyDict_New();

	if( !pyDetectNet_Type.tp_dict )
	{
		printf(LOG_PY_INFERENCE "detectNet failed to create new PyDict object\n");
		return false;
	}

	if( PyDict_SetItemString(pyDetectNet_Type.tp_dict, "Detection", (PyObject*)&pyDetection_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "detectNet failed to register detectNet.Detection inner class\n");
		return false;
	}

	// complete registration of the detectNet type
	if( PyType_Ready(&pyDetectNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "detectNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyDetectNet_Type);

	if( PyModule_AddObject(module, "detectNet", (PyObject*)&pyDetectNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "detectNet PyModule_AddObject('detectNet') failed\n");
		return false;
	}
	
	return true;
}


