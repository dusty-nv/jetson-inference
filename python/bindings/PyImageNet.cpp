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


typedef struct {
    PyTensorNet_Object base;
    imageNet* net;	// object instance
} PyImageNet_Object;


// Init
static int PyImageNet_Init( PyImageNet_Object* self, PyObject *args, PyObject *kwds )
{
	printf("PyImageNet_Init()\n");
	
	const char* network = "googlenet";
	static char* kwlist[] = {"network", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &network))
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- failed to parse args tuple in __init__()");
		printf("PyImageNet -- failed to parse args tuple in __init__()\n");
		return -1;
	}
    
	printf("PyImageNet -- loading build-in network '%s'\n", network);
	
	imageNet::NetworkType networkType = imageNet::NetworkTypeFromStr(network);
	
	if( networkType == imageNet::CUSTOM )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- invalid built-in network was requested");
		printf("PyImageNet -- invalid built-in network was requested ('%s')\n", network);
		return -1;
	}
	
	self->net = imageNet::Create(networkType);
	
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- failed to load network");
		printf("PyImageNet -- failed to load built-in network '%s'\n", network);
		return -1;
	}

	self->base.net = self->net;
    return 0;
}


// GetNetworkName
static PyObject* PyImageNet_GetNetworkName( PyImageNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- invalid object instance");
		return NULL;
	}
	
    return Py_BuildValue("s", self->net->GetNetworkName());
}


// GetNumClasses
static PyObject* PyImageNet_GetNumClasses( PyImageNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- invalid object instance");
		return NULL;
	}

#ifdef PYTHON_3
	return PyLong_FromUnsignedLong(self->net->GetNumClasses());
#else
    return PyInt_FromLong(self->net->GetNumClasses());
#endif
}


// GetClassDesc
PyObject* PyImageNet_GetClassDesc( PyImageNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassDesc(classIdx));
}


// GetClassSynset
PyObject* PyImageNet_GetClassSynset( PyImageNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- invalid object instance");
		return NULL;
	}
	
	int classIdx = 0;

	if( !PyArg_ParseTuple(args, "i", &classIdx) )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- failed to parse arguments");
		return NULL;
	}
		
	if( classIdx < 0 || classIdx >= self->net->GetNumClasses() )
	{
		PyErr_SetString(PyExc_Exception, "PyImageNet -- requested class index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetClassSynset(classIdx));
}

//-------------------------------------------------------------------------------
static PyTypeObject pyImageNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyImageNet_Methods[] = 
{
	{ "GetNetworkName", (PyCFunction)PyImageNet_GetNetworkName, METH_NOARGS, "Return the name of the build-in network used by the model, or 'custom' if using a custom-loaded model"},
    { "GetNumClasses", (PyCFunction)PyImageNet_GetNumClasses, METH_NOARGS, "Return the number of object classes that this network model is able to classify"},
	{ "GetClassDesc", (PyCFunction)PyImageNet_GetClassDesc, METH_VARARGS, "Return the class description for the given class index"},
	{ "GetClassSynset", (PyCFunction)PyImageNet_GetClassSynset, METH_VARARGS, "Return the class synset dataset category for the given class index"},
	{NULL}  /* Sentinel */
};

// Register type
bool PyImageNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	pyImageNet_Type.tp_name 	 = "jetson.inference.imageNet";
	pyImageNet_Type.tp_basicsize = sizeof(PyImageNet_Object);
	pyImageNet_Type.tp_flags 	 = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyImageNet_Type.tp_base      = PyTensorNet_Type();
	pyImageNet_Type.tp_methods   = pyImageNet_Methods;
	pyImageNet_Type.tp_new 		 = NULL; /*PyImageNet_New;*/
	pyImageNet_Type.tp_init		 = (initproc)PyImageNet_Init;
	pyImageNet_Type.tp_dealloc	 = NULL; /*(destructor)PyImageNet_Dealloc;*/
	pyImageNet_Type.tp_doc  	 = "Image Recognition DNN";
	 
	if( PyType_Ready(&pyImageNet_Type) < 0 )
	{
		printf("PyImageNet -- PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyImageNet_Type);
    
	if( PyModule_AddObject(module, "imageNet", (PyObject*)&pyImageNet_Type) < 0 )
	{
		printf("PyImageNet -- PyModule_AddObject('imageNet') failed\n");
		return false;
	}
	
	return true;
}