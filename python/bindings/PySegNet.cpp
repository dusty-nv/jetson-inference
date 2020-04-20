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
#include "PySegNet.h"

#include "segNet.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyTensorNet_Object base;
	segNet* net;	// object instance
} PySegNet_Object;


#define DOC_SEGNET   "Semantic Segmentation DNN - segments an image with per-pixel classification\n\n" \
				 "Examples (jetson-inference/python/examples)\n" \
                     "     segnet-console.py\n" \
				 "     segnet-camera.py\n\n" \
				 "__init__(...)\n" \
				 "     Loads an semantic segmentation model.\n\n" \
				 "     Parameters:\n" \
				 "       network (string) -- name of a built-in network to use,\n" \
				 "                           see below for available options.\n\n" \
				 "       argv (strings) -- command line arguments passed to segNet,\n" \
				 "                         see below for available options.\n\n" \
 				 SEGNET_USAGE_STRING


// Init
static int PySegNet_Init( PySegNet_Object* self, PyObject *args, PyObject *kwds )
{
	printf(LOG_PY_INFERENCE "PySegNet_Init()\n");

	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "fcn-resnet18-pascal-voc";
	static char* kwlist[] = {"network", "argv", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sO", kwlist, &network, &argList))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ failed to parse args tuple");
		printf(LOG_PY_INFERENCE "segNet.__init()__ failed to parse args tuple\n");
		return -1;
	}
    
	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		printf(LOG_PY_INFERENCE "segNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "segNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.__init()__ failed to parse argv list");
				return -1;
			}

			printf(LOG_PY_INFERENCE "segNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		self->net = segNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		printf(LOG_PY_INFERENCE "segNet loading build-in network '%s'\n", network);
		
		// parse the selected built-in network
		segNet::NetworkType networkType = segNet::NetworkTypeFromStr(network);
		
		if( networkType == segNet::SEGNET_CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid built-in network was requested");
			printf(LOG_PY_INFERENCE "segNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}
		
		// load the built-in network
		self->net = segNet::Create(networkType);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet failed to load network");
		printf(LOG_PY_INFERENCE "segNet failed to load built-in network '%s'\n", network);
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PySegNet_Dealloc( PySegNet_Object* self )
{
	printf(LOG_PY_INFERENCE "PySegNet_Dealloc()\n");

	// free the network
	if( self->net != NULL )
	{
		delete self->net;
		self->net = NULL;
	}

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_PROCESS  "Perform the initial inferencing processing of the segmentation.\n" \
                     "The results can then be visualized using the Overlay() and Mask() functions.\n\n" \
				 "Parameters:\n" \
				 "  image  (capsule) -- CUDA memory capsule\n" \
				 "  width  (int) -- width of the image (in pixels)\n" \
				 "  height (int) -- height of the image (in pixels)\n" \
				 "  ignore_class (string) -- optional label name of class to ignore in the classification (default: 'void')\n" \
				 "Returns:  (none)"

// Process
static PyObject* PySegNet_Process( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* ignore_class = "void";
	static char* kwlist[] = {"image", "width", "height", "ignore_class", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|s", kwlist, &capsule, &width, &height, &ignore_class))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to parse args tuple");
		printf(LOG_PY_INFERENCE "segNet.Process() failed to parse args tuple\n");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() image dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// classify the image
	const bool result = self->net->Process((float*)img, width, height, ignore_class);

	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Process() encountered an error segmenting the image");
		return NULL;
	}

	Py_RETURN_NONE;
}


#define DOC_OVERLAY  "Produce the segmentation overlay alpha blended on top of the original image.\n\n" \
				 "Parameters:\n" \
				 "  image  (capsule) -- output CUDA memory capsule\n" \
				 "  width  (int) -- width of the image (in pixels)\n" \
				 "  height (int) -- height of the image (in pixels)\n" \
				 "  filter_mode (string) -- optional string indicating the filter mode, 'point' or 'linear' (default: 'linear')\n" \
				 "Returns:  (none)"

// Overlay
static PyObject* PySegNet_Overlay( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* filter_str = "linear";
	static char* kwlist[] = {"image", "width", "height", "filter_mode", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|s", kwlist, &capsule, &width, &height, &filter_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() failed to parse args tuple");
		printf(LOG_PY_INFERENCE "segNet.Overlay() failed to parse args tuple\n");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() image dimensions are invalid");
		return NULL;
	}

	// verify filter mode
	segNet::FilterMode filterMode;

	if( strcmp(filter_str, "linear") == 0 )
		filterMode = segNet::FILTER_LINEAR;
	else if( strcmp(filter_str, "point") == 0 )
		filterMode = segNet::FILTER_POINT;
	else
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() invalid filter_mode parameter");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// visualize the image
	const bool result = self->net->Overlay((float*)img, width, height, filterMode);

	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Overlay() encountered an error segmenting the image");
		return NULL;
	}

	Py_RETURN_NONE;
}


#define DOC_MASK     "Produce a colorized RGBA segmentation mask of the output.\n\n" \
				 "Parameters:\n" \
				 "  image  (capsule) -- output CUDA memory capsule\n" \
				 "  width  (int) -- width of the image (in pixels)\n" \
				 "  height (int) -- height of the image (in pixels)\n" \
				 "  filter_mode (string) -- optional string indicating the filter mode, 'point' or 'linear' (default: 'linear')\n" \
				 "Returns:  (none)"

// Overlay
static PyObject* PySegNet_Mask( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* filter_str = "linear";
	static char* kwlist[] = {"image", "width", "height", "filter_mode", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii|s", kwlist, &capsule, &width, &height, &filter_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() failed to parse args tuple");
		printf(LOG_PY_INFERENCE "segNet.Mask() failed to parse args tuple\n");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() image dimensions are invalid");
		return NULL;
	}

	// verify filter mode
	segNet::FilterMode filterMode;

	if( strcmp(filter_str, "linear") == 0 )
		filterMode = segNet::FILTER_LINEAR;
	else if( strcmp(filter_str, "point") == 0 )
		filterMode = segNet::FILTER_POINT;
	else
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() invalid filter_mode parameter");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// visualize the image
	const bool result = self->net->Mask((float*)img, width, height, filterMode);

	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() encountered an error segmenting the image");
		return NULL;
	}

	Py_RETURN_NONE;
}


#define DOC_MASKCLASS   "Produce a grayscale binary segmentation mask, where the pixel values correspond to the class ID of the corresponding class type.\n\n" \
						"Parameters:\n" \
						"  image  (capsule) -- output CUDA memory capsule\n" \
						"  width  (int) -- width of the image (in pixels)\n" \
						"  height (int) -- height of the image (in pixels)\n" \
						"Returns:  (none)"

// Class Id Mask
static PyObject* PySegNet_MaskClass( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	static char* kwlist[] = {"image", "width", "height", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "Oii", kwlist, &capsule, &width, &height))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() failed to parse args tuple");
		printf(LOG_PY_INFERENCE "segNet.Mask() failed to parse args tuple\n");
		return NULL;
	}

	// verify dimensions
	if( width <= 0 || height <= 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() image dimensions are invalid");
		return NULL;
	}

	// get pointer to image data
	void* img = PyCUDA_GetPointer(capsule);

	if( !img )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() failed to get image pointer from PyCapsule container");
		return NULL;
	}

	// visualize the image
	const bool result = self->net->Mask((uint8_t*)img, width, height);

	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.Mask() encountered an error segmenting the image");
		return NULL;
	}

	Py_RETURN_NONE;
}


#define DOC_GET_NETWORK_NAME "Return the name of the built-in network used by the model.\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- name of the network (e.g. 'FCN_ResNet18', 'FCN_Alexnet')\n" \
					    "              or 'custom' if using a custom-loaded model"

// GetNetworkName
static PyObject* PySegNet_GetNetworkName( PySegNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	return Py_BuildValue("s", self->net->GetNetworkName());
}


#define DOC_GET_NUM_CLASSES "Return the number of object classes that this network model is able to classify.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (int) -- number of object classes that the model supports"

// GetNumClasses
static PyObject* PySegNet_GetNumClasses( PySegNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
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
PyObject* PySegNet_GetClassDesc( PySegNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassDesc(classIdx));
}


#define DOC_SET_OVERLAY_ALPHA "Set the alpha blending value used during overlay visualization for all classes\n\n" \
				 	  "Parameters:\n" \
					  "  alpha (float) -- desired alpha value, between 0.0 and 255.0\n" \
					  "  explicit_exempt (optional, bool) -- if True, the global alpha doesn't apply to classes that have an alpha value explicitly set in the colors file (default: True)\n" \
					  "Returns:  (none)"

// SetOverlayAlpha
PyObject* PySegNet_SetOverlayAlpha( PySegNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet invalid object instance");
		return NULL;
	}
	
	float alpha = 0.0f;
	int exempt = 1;

	static char* kwlist[] = {"alpha", "explicit_exempt", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f|i", kwlist, &alpha, &exempt) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.SetOverlayAlpha() failed to parse arguments");
		return NULL;
	}
		
	if( alpha < 0.0f || alpha > 255.0f )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "segNet.SetOverlayAlpha() -- provided alpha value is out-of-range");
		return NULL;
	}

	const bool explicit_exempt = (exempt <= 0) ? false : true;

	self->net->SetOverlayAlpha(alpha, explicit_exempt);

	Py_RETURN_NONE;
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PySegNet_Usage( PySegNet_Object* self )
{
	return Py_BuildValue("s", segNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject PySegNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef PySegNet_Methods[] = 
{
	{ "Process", (PyCFunction)PySegNet_Process, METH_VARARGS|METH_KEYWORDS, DOC_PROCESS},
	{ "Overlay", (PyCFunction)PySegNet_Overlay, METH_VARARGS|METH_KEYWORDS, DOC_OVERLAY},
	{ "Mask", (PyCFunction)PySegNet_Mask, METH_VARARGS|METH_KEYWORDS, DOC_MASK},
	{ "MaskClass", (PyCFunction)PySegNet_MaskClass, METH_VARARGS|METH_KEYWORDS, DOC_MASKCLASS},
	{ "GetNetworkName", (PyCFunction)PySegNet_GetNetworkName, METH_NOARGS, DOC_GET_NETWORK_NAME},
     { "GetNumClasses", (PyCFunction)PySegNet_GetNumClasses, METH_NOARGS, DOC_GET_NUM_CLASSES},
	{ "GetClassDesc", (PyCFunction)PySegNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "SetOverlayAlpha", (PyCFunction)PySegNet_SetOverlayAlpha, METH_VARARGS|METH_KEYWORDS, DOC_SET_OVERLAY_ALPHA},
	{ "Usage", (PyCFunction)PySegNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},
	{NULL}  /* Sentinel */
};

// Register type
bool PySegNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	PySegNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".segNet";
	PySegNet_Type.tp_basicsize	= sizeof(PySegNet_Object);
	PySegNet_Type.tp_flags		= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	PySegNet_Type.tp_base		= PyTensorNet_Type();
	PySegNet_Type.tp_methods		= PySegNet_Methods;
	PySegNet_Type.tp_new		= NULL; /*PySegNet_New;*/
	PySegNet_Type.tp_init		= (initproc)PySegNet_Init;
	PySegNet_Type.tp_dealloc		= (destructor)PySegNet_Dealloc;
	PySegNet_Type.tp_doc		= DOC_SEGNET;
	 
	if( PyType_Ready(&PySegNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "segNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&PySegNet_Type);
    
	if( PyModule_AddObject(module, "segNet", (PyObject*)&PySegNet_Type) < 0 )
	{
		printf(LOG_PY_INFERENCE "segNet PyModule_AddObject('imageNet') failed\n");
		return false;
	}
	
	return true;
}
