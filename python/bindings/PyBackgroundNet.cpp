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
#include "PyBackgroundNet.h"

#include "backgroundNet.h"
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyTensorNet_Object base;
	backgroundNet* net;		// object instance
} PyBackgroundNet_Object;


#define DOC_BACKGROUNDNET "Background removal DNN - performs background subtraction on images.\n\n" \
				 "Examples (jetson-inference/python/examples)\n" \
                     "     backgroundnet.py\n\n" \
				 "__init__(...)\n" \
				 "     Loads a background subtraction/removal model.\n\n" \
				 "     Parameters:\n" \
				 "       network (string) -- name of a built-in network to use,\n" \
				 "                           see below for available options.\n\n" \
				 "       argv (strings) -- command line arguments passed to backgroundNet,\n" \
				 "                         see below for available options.\n\n" \
				 "     Extended parameters for loading custom models:\n" \
				 "       model (string) -- path to self-trained ONNX model to load.\n\n" \
				 "       input_blob (string) -- name of the input layer of the model.\n\n" \
				 "       output_blob (string) -- name of the output layer of the model.\n\n" \
 				 BACKGROUNDNET_USAGE_STRING

// Init
static int PyBackgroundNet_Init( PyBackgroundNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyBackgroundNet_Init()\n");

	// parse arguments
	PyObject* argList = NULL;
	
	const char* network     = "u2net";
	const char* model       = NULL;
	const char* input_blob  = BACKGROUNDNET_DEFAULT_INPUT;
	const char* output_blob = BACKGROUNDNET_DEFAULT_OUTPUT;
	
	static char* kwlist[] = {"network", "argv", "model", "input_blob", "output_blob", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOsss", kwlist, &network, &argList, &model, &input_blob, &output_blob))
		return -1;

	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		LogDebug(LOG_PY_INFERENCE "backgroundNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "backgroundNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "backgroundNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		Py_BEGIN_ALLOW_THREADS
		self->net = backgroundNet::Create(argc, argv);
		Py_END_ALLOW_THREADS
		
		// free the arguments array
		free(argv);
	}
	else
	{
		LogDebug(LOG_PY_INFERENCE "backgroundNet loading custom model '%s'\n", model);
		
		// load the network using custom model parameters
		Py_BEGIN_ALLOW_THREADS
		self->net = backgroundNet::Create(model != NULL ? model : network, input_blob, output_blob);
		Py_END_ALLOW_THREADS
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet failed to load network");
		return -1;
	}
	
	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyBackgroundNet_Dealloc( PyBackgroundNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyBackgroundNet_Dealloc()\n");

	// free the network
	SAFE_DELETE(self->net);

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_PROCESS  "Perform background subtraction/removal on the image.\n" \
                     "If only an input image is provided, the operation will be performed in-place.\n\n" \
				 "If an output image is also provided, the results will be written to the output image.\n" \
				 "If the images have an alpha channel (uchar4 or float4) and mask_alpha is true (default),\n" \
				 "then the background/foreground mask will be written to the output's alpha channel.\n\n" \
				 "Parameters:\n" \
				 "  input  (capsule) -- CUDA memory capsule (input image)\n" \
				 "  output (capsule) -- CUDA memory capsule (optional output image)\n" \
				 "  filter (string)  -- filtering used in upscaling the mask, 'point' or 'linear' (default is 'linear')\n" \
				 "  mask_alpha (bool) -- if true (default), the mask will be applied to the alpha channel as well\n" \
				 "Returns:  (none)"

// Process
static PyObject* PyBackgroundNet_Process( PyBackgroundNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* input_capsule = NULL;
	PyObject* output_capsule = NULL;
	
	const char* filter_str = "linear";
	int mask_alpha_int = 1;
	
	static char* kwlist[] = {"input", "output", "filter", "mask_alpha", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|Ossi", kwlist, &input_capsule, &output_capsule, &filter_str, &mask_alpha_int))
		return NULL;

	const bool mask_alpha = (mask_alpha_int <= 0) ? false : true;
	const cudaFilterMode filter_mode = cudaFilterModeFromStr(filter_str);
	
	// get pointers to image data
	PyCudaImage* input_img = PyCUDA_GetImage(input_capsule);
	
	if( !input_img ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from input argument");
		return NULL;
	}
	
	if( output_capsule != NULL )
	{
		// get pointers to image data
		PyCudaImage* output_img = PyCUDA_GetImage(output_capsule);
		
		if( !output_img ) 
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from output argument");
			return NULL;
		}
		
		if( input_img->width != output_img->width || input_img->height != output_img->height )
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "input and output image dimensions don't match");
			return NULL;
		}
		
		if( input_img->format != output_img->format )
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "input and output image formats don't match");
			return NULL;
		}
			
		bool result = false;
		Py_BEGIN_ALLOW_THREADS
		
		result = self->net->Process(input_img->base.ptr, output_img->base.ptr, input_img->width, input_img->height, input_img->format,
							   filter_mode, mask_alpha);
							   
		Py_END_ALLOW_THREADS
		
		if( !result )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet.Process() encountered an error processing the image");
			return NULL;
		}
	}
	else
	{
		bool result = false;
		Py_BEGIN_ALLOW_THREADS
		result = self->net->Process(input_img->base.ptr, input_img->width, input_img->height, input_img->format, filter_mode, mask_alpha);
		Py_END_ALLOW_THREADS
				
		if( !result )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet.Process() encountered an error processing the image");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


#define DOC_GET_NETWORK_NAME "Return the name of the built-in network used by the model.\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- name of the network (e.g. 'u2net')\n" \
					    "              or 'custom' if using a custom-loaded model"

// GetNetworkName
static PyObject* PyBackgroundNet_GetNetworkName( PyBackgroundNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "backgroundNet invalid object instance");
		return NULL;
	}
	
	return Py_BuildValue("s", self->net->GetNetworkName());
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyBackgroundNet_Usage( PyBackgroundNet_Object* self )
{
	return Py_BuildValue("s", backgroundNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject PyBackgroundNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef PyBackgroundNet_Methods[] = 
{
	{ "Process", (PyCFunction)PyBackgroundNet_Process, METH_VARARGS|METH_KEYWORDS, DOC_PROCESS},
	{ "GetNetworkName", (PyCFunction)PyBackgroundNet_GetNetworkName, METH_NOARGS, DOC_GET_NETWORK_NAME},
	{ "Usage", (PyCFunction)PyBackgroundNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},
	{NULL}  /* Sentinel */
};

// Register type
bool PyBackgroundNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	PyBackgroundNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".backgroundNet";
	PyBackgroundNet_Type.tp_basicsize	= sizeof(PyBackgroundNet_Object);
	PyBackgroundNet_Type.tp_flags		= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	PyBackgroundNet_Type.tp_base		= PyTensorNet_Type();
	PyBackgroundNet_Type.tp_methods	= PyBackgroundNet_Methods;
	PyBackgroundNet_Type.tp_new		= NULL; /*PyBackgroundNet_New;*/
	PyBackgroundNet_Type.tp_init		= (initproc)PyBackgroundNet_Init;
	PyBackgroundNet_Type.tp_dealloc	= (destructor)PyBackgroundNet_Dealloc;
	PyBackgroundNet_Type.tp_doc		= DOC_BACKGROUNDNET;
	 
	if( PyType_Ready(&PyBackgroundNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "backgroundNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&PyBackgroundNet_Type);
    
	if( PyModule_AddObject(module, "backgroundNet", (PyObject*)&PyBackgroundNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "backgroundNet PyModule_AddObject('backgroundNet') failed\n");
		return false;
	}
	
	return true;
}
