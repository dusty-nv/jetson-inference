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
#include "PyImageNet.h"

#include "imageNet.h"
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
    PyTensorNet_Object base;
    imageNet* net;	// object instance
} PyImageNet_Object;


#define DOC_IMAGENET "Image Recognition DNN - classifies an image\n\n" \
				 "Examples (jetson-inference/python/examples)\n" \
				 "     my-recognition.py\n" \
                     "     imagenet-console.py\n" \
				 "     imagenet-camera.py\n\n" \
				 "__init__(...)\n" \
				 "     Loads an image recognition model.\n\n" \
				 "     Parameters:\n" \
				 "       network (string) -- name of a built-in network to use,\n" \
				 "                           see below for available options.\n\n" \
				 "       argv (strings) -- command line arguments passed to imageNet,\n" \
				 "                         see below for available options.\n\n" \
 				 IMAGENET_USAGE_STRING


// Init
static int PyImageNet_Init( PyImageNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyImageNet_Init()\n");
	
	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "googlenet";
	static char* kwlist[] = {"network", "argv", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sO", kwlist, &network, &argList))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet.__init()__ failed to parse args tuple");
		printf(LOG_PY_INFERENCE "imageNet.__init()__ failed to parse args tuple\n");
		return -1;
	}
    
	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		printf(LOG_PY_INFERENCE "imageNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "imageNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "imageNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		self->net = imageNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		printf(LOG_PY_INFERENCE "imageNet loading build-in network '%s'\n", network);
		
		// parse the selected built-in network
		imageNet::NetworkType networkType = imageNet::NetworkTypeFromStr(network);
		
		if( networkType == imageNet::CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid built-in network was requested");
			printf(LOG_PY_INFERENCE "imageNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}
		
		// load the built-in network
		self->net = imageNet::Create(networkType);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet failed to load network");
		printf(LOG_PY_INFERENCE "imageNet failed to load built-in network '%s'\n", network);
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyImageNet_Dealloc( PyImageNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyImageNet_Dealloc()\n");

	// delete the network
	SAFE_DELETE(self->net);
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_CLASSIFY "Classify an RGBA image and return the object's class and confidence.\n\n" \
				 "Parameters:\n" \
				 "  image  (capsule) -- CUDA memory capsule\n" \
				 "  width  (int) -- width of the image (in pixels)\n" \
				 "  height (int) -- height of the image (in pixels)\n\n" \
				 "Returns:\n" \
				 "  (int, float) -- tuple containing the object's class index and confidence"

// Classify
static PyObject* PyImageNet_Classify( PyImageNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* format_str = "rgba32f";
	static char* kwlist[] = {"image", "width", "height", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iis", kwlist, &capsule, &width, &height, &format_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet.Classify() failed to parse args tuple");
		printf(LOG_PY_INFERENCE "imageNet.Classify() failed to parse args tuple\n");
		return NULL;
	}

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// classify the image
	float confidence = 0.0f;

	const int img_class = self->net->Classify(ptr, width, height, format, &confidence);

	if( img_class < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet.Classify() encountered an error classifying the image");
		return NULL;
	}

	// create output objects
	PyObject* pyClass = PYLONG_FROM_LONG(img_class);
	PyObject* pyConf  = PyFloat_FromDouble(confidence);

	// return tuple
	PyObject* tuple = PyTuple_Pack(2, pyClass, pyConf);

	Py_DECREF(pyClass);
	Py_DECREF(pyConf);

	return tuple;
    
}


#define DOC_GET_NETWORK_NAME "Return the name of the built-in network used by the model.\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- name of the network (e.g. 'googlenet', 'alexnet')\n" \
					    "              or 'custom' if using a custom-loaded model"

// GetNetworkName
static PyObject* PyImageNet_GetNetworkName( PyImageNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid object instance");
		return NULL;
	}
	
	return Py_BuildValue("s", self->net->GetNetworkName());
}


#define DOC_GET_NUM_CLASSES "Return the number of object classes that this network model is able to classify.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (int) -- number of object classes that the model supports"

// GetNumClasses
static PyObject* PyImageNet_GetNumClasses( PyImageNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid object instance");
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
PyObject* PyImageNet_GetClassDesc( PyImageNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet requested class index is out of bounds");
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
PyObject* PyImageNet_GetClassSynset( PyImageNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "imageNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassSynset(classIdx));
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyImageNet_Usage( PyImageNet_Object* self )
{
	return Py_BuildValue("s", imageNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject pyImageNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyImageNet_Methods[] = 
{
	{ "Classify", (PyCFunction)PyImageNet_Classify, METH_VARARGS|METH_KEYWORDS, DOC_CLASSIFY},
	{ "GetNetworkName", (PyCFunction)PyImageNet_GetNetworkName, METH_NOARGS, DOC_GET_NETWORK_NAME},
     { "GetNumClasses", (PyCFunction)PyImageNet_GetNumClasses, METH_NOARGS, DOC_GET_NUM_CLASSES},
	{ "GetClassDesc", (PyCFunction)PyImageNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetClassSynset", (PyCFunction)PyImageNet_GetClassSynset, METH_VARARGS, DOC_GET_CLASS_SYNSET},
	{ "Usage", (PyCFunction)PyImageNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},
	{NULL}  /* Sentinel */
};

// Register type
bool PyImageNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	pyImageNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".imageNet";
	pyImageNet_Type.tp_basicsize	= sizeof(PyImageNet_Object);
	pyImageNet_Type.tp_flags		= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyImageNet_Type.tp_base		= PyTensorNet_Type();
	pyImageNet_Type.tp_methods	= pyImageNet_Methods;
	pyImageNet_Type.tp_new		= NULL; /*PyImageNet_New;*/
	pyImageNet_Type.tp_init		= (initproc)PyImageNet_Init;
	pyImageNet_Type.tp_dealloc	= (destructor)PyImageNet_Dealloc;
	pyImageNet_Type.tp_doc		= DOC_IMAGENET;
	 
	if( PyType_Ready(&pyImageNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "imageNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyImageNet_Type);
    
	if( PyModule_AddObject(module, "imageNet", (PyObject*)&pyImageNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "imageNet PyModule_AddObject('imageNet') failed\n");
		return false;
	}
	
	return true;
}
