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

#include "PyTensorNet.h"
#include "PyActionNet.h"

#include "actionNet.h"
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
    PyTensorNet_Object base;
    actionNet* net;	// object instance
} PyActionNet_Object;


#define DOC_IMAGENET "Action Recognition DNN - classifies an image sequence\n\n" \
				 "Examples (jetson-inference/python/examples)\n" \
				 "     actionnet.py\n" \
				 "__init__(...)\n" \
				 "     Loads an action classification model.\n\n" \
				 "     Parameters:\n" \
				 "       network (string) -- name of a built-in network to use,\n" \
				 "                           see below for available options.\n\n" \
				 "       argv (strings) -- command line arguments passed to actionNet,\n" \
				 "                         see below for available options.\n\n" \
				 "     Extended parameters for loading custom models:\n" \
				 "       model (string) -- path to self-trained ONNX model to load.\n\n" \
				 "       labels (string) -- path to labels.txt file (optional)\n\n" \
				 "       input_blob (string) -- name of the input layer of the model.\n\n" \
				 "       output_blob (string) -- name of the output layer of the model.\n\n" \
 				 ACTIONNET_USAGE_STRING

// Init
static int PyActionNet_Init( PyActionNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyActionNet_Init()\n");
	
	// parse arguments
	PyObject* argList = NULL;
	
	const char* network     = "resnet-18";
	const char* model       = NULL;
	const char* labels      = NULL;
	const char* input_blob  = ACTIONNET_DEFAULT_INPUT;
	const char* output_blob = ACTIONNET_DEFAULT_OUTPUT;
	
	static char* kwlist[] = {"network", "argv", "model", "labels", "input_blob", "output_blob", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOssss", kwlist, &network, &argList, &model, &labels, &input_blob, &output_blob))
		return -1;

	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		LogDebug(LOG_PY_INFERENCE "actionNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "actionNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "actionNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		Py_BEGIN_ALLOW_THREADS
		self->net = actionNet::Create(argc, argv);
		Py_END_ALLOW_THREADS
		
		// free the arguments array
		free(argv);
	}
	else
	{
		LogDebug(LOG_PY_INFERENCE "actionNet loading custom model '%s'\n", model);
		
		// load the network using custom model parameters
		Py_BEGIN_ALLOW_THREADS
		self->net = actionNet::Create(model != NULL ? model : network, labels, input_blob, output_blob);
		Py_END_ALLOW_THREADS
	}
	
	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet failed to load network");
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyActionNet_Dealloc( PyActionNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyActionNet_Dealloc()\n");

	// delete the network
	SAFE_DELETE(self->net);
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_CLASSIFY "Append an image to the sequence and classify the action, returning the class and confidence.\n\n" \
				 "Parameters:\n" \
				 "  image  (capsule) -- CUDA memory capsule\n" \
				 "  width  (int) -- width of the image (in pixels)\n" \
				 "  height (int) -- height of the image (in pixels)\n\n" \
				 "Returns:\n" \
				 "  (int, float) -- tuple containing the action's class index and confidence"

// Classify
static PyObject* PyActionNet_Classify( PyActionNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	int width = 0;
	int height = 0;

	const char* format_str = "rgba32f";
	static char* kwlist[] = {"image", "width", "height", "format", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|iis", kwlist, &capsule, &width, &height, &format_str))
		return NULL;

	// parse format string
	imageFormat format = imageFormatFromStr(format_str);

	// get pointer to image data
	void* ptr = PyCUDA_GetImage(capsule, &width, &height, &format);

	if( !ptr )
		return NULL;

	// classify the image
	float confidence = 0.0f;
	int img_class = -1;
	
	Py_BEGIN_ALLOW_THREADS
	img_class = self->net->Classify(ptr, width, height, format, &confidence);
	Py_END_ALLOW_THREADS
	
	if( img_class < -1 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet.Classify() encountered an error classifying the image");
		return NULL;
	}

	// create output objects
	PyObject* pyClass = PYLONG_FROM_LONG(img_class);
	PyObject* pyConf  = PyFloat_FromDouble(confidence);
	PyObject* tuple   = PyTuple_Pack(2, pyClass, pyConf);

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
static PyObject* PyActionNet_GetNetworkName( PyActionNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}
	
	return Py_BuildValue("s", self->net->GetNetworkName());
}


#define DOC_GET_NUM_CLASSES "Return the number of object classes that this network model is able to classify.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (int) -- number of object classes that the model supports"

// GetNumClasses
static PyObject* PyActionNet_GetNumClasses( PyActionNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
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
PyObject* PyActionNet_GetClassDesc( PyActionNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
		return NULL;

	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassDesc(classIdx));
}


#define DOC_GET_THRESHOLD  "Return the minimum confidence threshold for classification.\n\n" \
					  "Parameters:  (none)\n\n" \
					  "Returns:\n" \
					  "  (float) -- the confidence threshold for classification"

// GetThreshold
static PyObject* PyActionNet_GetThreshold( PyActionNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->net->GetThreshold());
}


#define DOC_SET_THRESHOLD  "Set the minimum confidence threshold for classification.\n\n" \
					  "Parameters:\n" \
					  "  (float) -- confidence threshold\n\n" \
					  "Returns:  (none)"

// SetThreshold
PyObject* PyActionNet_SetThreshold( PyActionNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}
	
	float threshold = 0.0f;

	if( !PyArg_ParseTuple(args, "f", &threshold) )
		return NULL;

	self->net->SetThreshold(threshold);
	Py_RETURN_NONE;
}


#define DOC_GET_SKIP_FRAMES "Return the number of frames that are skipped in between classifications.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (int) -- the number of frames skipped in between classifications"

// GetSkipFrames
static PyObject* PyActionNet_GetSkipFrames( PyActionNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->net->GetSkipFrames());
}


#define DOC_SET_SKIP_FRAMES  	"Set the number of frames that are skipped in between classifications.\n" \
						"Since actionNet operates on video sequences, it's often helpful to skip frames\n" \
						"to lengthen the window of time the model gets to 'see' an action being performed.\n\n" \
						"The default setting is 1, where every other frame is skipped.\n" \
						"Setting this to 0 will disable it, and every frame will be processed.\n" \
						"When a frame is skipped, the classification results from the last frame are returned.\n\n" \
						"Parameters:\n" \
						"  (int) -- the number of frames skipped in between classifications\n\n" \
						"Returns:  (none)"

// SetSkipFrames
PyObject* PyActionNet_SetSkipFrames( PyActionNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "actionNet invalid object instance");
		return NULL;
	}
	
	int skipFrames = 0;

	if( !PyArg_ParseTuple(args, "i", &skipFrames) )
		return NULL;

	self->net->SetSkipFrames(skipFrames);
	Py_RETURN_NONE;
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyActionNet_Usage( PyActionNet_Object* self )
{
	return Py_BuildValue("s", actionNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject pyActionNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyActionNet_Methods[] = 
{
	{ "Classify", (PyCFunction)PyActionNet_Classify, METH_VARARGS|METH_KEYWORDS, DOC_CLASSIFY},
	{ "GetNetworkName", (PyCFunction)PyActionNet_GetNetworkName, METH_NOARGS, DOC_GET_NETWORK_NAME},
     { "GetNumClasses", (PyCFunction)PyActionNet_GetNumClasses, METH_NOARGS, DOC_GET_NUM_CLASSES},
	{ "GetClassLabel", (PyCFunction)PyActionNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetClassDesc", (PyCFunction)PyActionNet_GetClassDesc, METH_VARARGS, DOC_GET_CLASS_DESC},
	{ "GetThreshold", (PyCFunction)PyActionNet_GetThreshold, METH_NOARGS, DOC_GET_THRESHOLD},
	{ "SetThreshold", (PyCFunction)PyActionNet_SetThreshold, METH_VARARGS, DOC_SET_THRESHOLD},
	{ "GetSkipFrames", (PyCFunction)PyActionNet_GetSkipFrames, METH_NOARGS, DOC_GET_SKIP_FRAMES},
	{ "SetSkipFrames", (PyCFunction)PyActionNet_SetSkipFrames, METH_VARARGS, DOC_SET_SKIP_FRAMES},
	{ "Usage", (PyCFunction)PyActionNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},
	{NULL}  /* Sentinel */
};

// Register type
bool PyActionNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	pyActionNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".actionNet";
	pyActionNet_Type.tp_basicsize	= sizeof(PyActionNet_Object);
	pyActionNet_Type.tp_flags		= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyActionNet_Type.tp_base		= PyTensorNet_Type();
	pyActionNet_Type.tp_methods	= pyActionNet_Methods;
	pyActionNet_Type.tp_new		= NULL; /*PyActionNet_New;*/
	pyActionNet_Type.tp_init		= (initproc)PyActionNet_Init;
	pyActionNet_Type.tp_dealloc	= (destructor)PyActionNet_Dealloc;
	pyActionNet_Type.tp_doc		= DOC_IMAGENET;
	 
	if( PyType_Ready(&pyActionNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "actionNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyActionNet_Type);
    
	if( PyModule_AddObject(module, "actionNet", (PyObject*)&pyActionNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "actionNet PyModule_AddObject('actionNet') failed\n");
		return false;
	}
	
	return true;
}
