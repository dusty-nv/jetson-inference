/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
#include "PyDepthNet.h"

#include "depthNet.h"
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


typedef struct {
	PyTensorNet_Object base;
	depthNet* net;			// object instance
	PyObject* depthField;	// depth field cudaImage
} PyDepthNet_Object;


#define DOC_DEPTHNET "Mono depth estimation DNN - performs depth mapping on monocular images\n\n" \
				 "Examples (jetson-inference/python/examples)\n" \
                     "     depthnet.py\n\n" \
				 "__init__(...)\n" \
				 "     Loads an semantic segmentation model.\n\n" \
				 "     Parameters:\n" \
				 "       network (string) -- name of a built-in network to use,\n" \
				 "                           see below for available options.\n\n" \
				 "       argv (strings) -- command line arguments passed to depthNet,\n" \
				 "                         see below for available options.\n\n" \
 				 DEPTHNET_USAGE_STRING


// Init
static int PyDepthNet_Init( PyDepthNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyDepthNet_Init()\n");

	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "fcn-mobilenet";
	static char* kwlist[] = {"network", "argv", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sO", kwlist, &network, &argList))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.__init()__ failed to parse arguments");
		LogError(LOG_PY_INFERENCE "depthNet.__init()__ failed to parse arguments\n");
		return -1;
	}
    
	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		LogVerbose(LOG_PY_INFERENCE "depthNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "depthNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "depthNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		self->net = depthNet::Create(argc, argv);

		// free the arguments array
		free(argv);
	}
	else
	{
		LogVerbose(LOG_PY_INFERENCE "depthNet loading build-in network '%s'\n", network);
		
		// parse the selected built-in network
		depthNet::NetworkType networkType = depthNet::NetworkTypeFromStr(network);
		
		if( networkType == depthNet::CUSTOM )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid built-in network was requested");
			LogError(LOG_PY_INFERENCE "depthNet invalid built-in network was requested ('%s')\n", network);
			return -1;
		}
		
		// load the built-in network
		self->net = depthNet::Create(networkType);
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet failed to load network");
		LogError(LOG_PY_INFERENCE "depthNet failed to load network\n");
		return -1;
	}

	// create an image capsule for the depth field
	self->depthField = PyCUDA_RegisterImage(self->net->GetDepthField(), self->net->GetDepthFieldWidth(), self->net->GetDepthFieldHeight(),
									IMAGE_GRAY32F, true, false);
	
	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyDepthNet_Dealloc( PyDepthNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyDepthNet_Dealloc()\n");

	// free the network
	SAFE_DELETE(self->net);

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_PROCESS  "Compute the depth field from a monocular RGB/RGBA image.\n" \
                     "The results can also be visualized if output image is provided.\n\n" \
				 "Parameters:\n" \
				 "  input  (capsule) -- CUDA memory capsule (input image)\n" \
				 "  output (capsule) -- CUDA memory capsule (optional output image)\n" \
				 "  colormap (string) -- colormap name (optional)\n" \
				 "  filter_mode (string) -- filtering used in upscaling, 'point' or 'linear' (default is 'linear')\n" \
				 "Returns:  (none)"

// Process
static PyObject* PyDepthNet_Process( PyDepthNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* input_capsule = NULL;
	PyObject* output_capsule = NULL;
	
	const char* colormap_str = "viridis";
	const char* filter_str = "linear";

	static char* kwlist[] = {"input", "output", "colormap", "filter", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|Oss", kwlist, &input_capsule, &output_capsule, &colormap_str, &filter_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.Process() failed to parse args tuple");
		return NULL;
	}

	// get pointers to image data
	PyCudaImage* input_img = PyCUDA_GetImage(input_capsule);
	
	if( !input_img ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from input argument");
		return NULL;
	}
	
	if( output_capsule != NULL )
	{
		const cudaColormapType colormap = cudaColormapFromStr(colormap_str);
		const cudaFilterMode filterMode = cudaFilterModeFromStr(filter_str);

		// get pointers to image data
		PyCudaImage* output_img = PyCUDA_GetImage(output_capsule);
		
		if( !output_img ) 
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from output argument");
			return NULL;
		}
		
		const bool result = self->net->Process(input_img->base.ptr, input_img->width, input_img->height, input_img->format,
									    output_img->base.ptr, output_img->width, output_img->height, output_img->format,
									    colormap, filterMode);
									    
		if( !result )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.Process() encountered an error processing the image");
			return NULL;
		}
	}
	else
	{
		const bool result = self->net->Process(input_img->base.ptr, input_img->width, input_img->height, input_img->format);
									    
		if( !result )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.Process() encountered an error processing the image");
			return NULL;
		}
	}

	Py_RETURN_NONE;
}


#define DOC_VISUALIZE "Visualize the raw depth field into a colorized RGB/RGBA depth map.\n\n" \
				  "Parameters:\n" \
				  "  output (capsule) -- output CUDA memory capsule\n" \
				  "  colormap (string) -- colormap name (optional)\n" \
				  "  filter_mode (string) -- filtering used in upscaling, 'point' or 'linear' (default is 'linear')\n" \
				  "Returns:  (none)"

// Visualize
static PyObject* PyDepthNet_Visualize( PyDepthNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* output_capsule = NULL;
	
	const char* colormap_str = "viridis";
	const char* filter_str = "linear";

	static char* kwlist[] = {"output", "colormap", "filter", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|ss", kwlist, &output_capsule, &colormap_str, &filter_str))
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.Process() failed to parse args tuple");
		return NULL;
	}

	// parse flags
	const cudaColormapType colormap = cudaColormapFromStr(colormap_str);
	const cudaFilterMode filterMode = cudaFilterModeFromStr(filter_str);

	// get pointers to image data
	PyCudaImage* output_img = PyCUDA_GetImage(output_capsule);
	
	if( !output_img ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from output argument");
		return NULL;
	}
		
	const bool result = self->net->Visualize(output_img->base.ptr, output_img->width, output_img->height, output_img->format,
								      colormap, filterMode);
								    
	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet.Visualize() encountered an error processing the image");
		return NULL;
	}

	Py_RETURN_NONE;
}


#define DOC_GET_NETWORK_NAME "Return the name of the built-in network used by the model.\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- name of the network (e.g. 'MonoDepth-Mobilenet', 'MonoDepth-ResNet18')\n" \
					    "              or 'custom' if using a custom-loaded model"

// GetNetworkName
static PyObject* PyDepthNet_GetNetworkName( PyDepthNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}
	
	return Py_BuildValue("s", self->net->GetNetworkName());
}


#define DOC_GET_DEPTH_FIELD "Return a cudaImage object of the raw depth field.\n" \
					   "This is a single-channel float32 image that contains the depth estimates.\n\n" \
				 	   "Parameters:  (none)\n\n" \
					   "Returns:\n" \
					   "  (cudaImage) -- single-channel float32 depth field"

// GetDepthField
static PyObject* PyDepthNet_GetDepthField( PyDepthNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}

	Py_INCREF(self->depthField);
	return self->depthField;
}


#define DOC_GET_DEPTH_FIELD_WIDTH  "Return the width of the depth field, in pixels.\n\n" \
					          "Parameters:  (none)\n\n" \
					          "Returns:\n" \
					          "  (int) -- width of the depth field, in pixels" \

// GetDepthFieldWidth
static PyObject* PyDepthNet_GetDepthFieldWidth( PyDepthNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->net->GetDepthFieldWidth());
}


#define DOC_GET_DEPTH_FIELD_HEIGHT "Return the height of the depth field, in pixels.\n\n" \
					          "Parameters:  (none)\n\n" \
					          "Returns:\n" \
					          "  (int) -- height of the depth field, in pixels" \

// GetDepthFieldHeight
static PyObject* PyDepthNet_GetDepthFieldHeight( PyDepthNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "depthNet invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->net->GetDepthFieldHeight());
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyDepthNet_Usage( PyDepthNet_Object* self )
{
	return Py_BuildValue("s", depthNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject PyDepthNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef PyDepthNet_Methods[] = 
{
	{ "Process", (PyCFunction)PyDepthNet_Process, METH_VARARGS|METH_KEYWORDS, DOC_PROCESS},
	{ "Visualize", (PyCFunction)PyDepthNet_Visualize, METH_VARARGS|METH_KEYWORDS, DOC_VISUALIZE},
	{ "GetNetworkName", (PyCFunction)PyDepthNet_GetNetworkName, METH_NOARGS, DOC_GET_NETWORK_NAME},
     { "GetDepthField", (PyCFunction)PyDepthNet_GetDepthField, METH_NOARGS, DOC_GET_DEPTH_FIELD},
	{ "GetDepthFieldWidth", (PyCFunction)PyDepthNet_GetDepthFieldWidth, METH_NOARGS, DOC_GET_DEPTH_FIELD_WIDTH},
	{ "GetDepthFieldHeight", (PyCFunction)PyDepthNet_GetDepthFieldHeight, METH_NOARGS, DOC_GET_DEPTH_FIELD_HEIGHT},
	{ "Usage", (PyCFunction)PyDepthNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},
	{NULL}  /* Sentinel */
};

// Register type
bool PyDepthNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	PyDepthNet_Type.tp_name		= PY_INFERENCE_MODULE_NAME ".depthNet";
	PyDepthNet_Type.tp_basicsize	= sizeof(PyDepthNet_Object);
	PyDepthNet_Type.tp_flags		= Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	PyDepthNet_Type.tp_base		= PyTensorNet_Type();
	PyDepthNet_Type.tp_methods	= PyDepthNet_Methods;
	PyDepthNet_Type.tp_new		= NULL; /*PyDepthNet_New;*/
	PyDepthNet_Type.tp_init		= (initproc)PyDepthNet_Init;
	PyDepthNet_Type.tp_dealloc	= (destructor)PyDepthNet_Dealloc;
	PyDepthNet_Type.tp_doc		= DOC_DEPTHNET;
	 
	if( PyType_Ready(&PyDepthNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "depthNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&PyDepthNet_Type);
    
	if( PyModule_AddObject(module, "depthNet", (PyObject*)&PyDepthNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "depthNet PyModule_AddObject('depthNet') failed\n");
		return false;
	}
	
	return true;
}
