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

#include "tensorNet.h"
#include "logging.h"


// New
static PyObject* PyTensorNet_New( PyTypeObject *type, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyTensorNet_New()\n");
	
	// allocate a new container
	PyTensorNet_Object* self = (PyTensorNet_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "tensorNet tp_alloc() failed to allocate a new object");
		LogError(LOG_PY_INFERENCE "tensorNet tp_alloc() failed to allocate a new object\n");
		return NULL;
	}
	
	self->net = NULL;
	return (PyObject*)self;
}


// Deallocate
static void PyTensorNet_Dealloc( PyTensorNet_Object* self )
{
	LogDebug("PyTensorNet_Dealloc()\n");

	// free the network
	if( self->net != NULL )
	{
		delete self->net;
		self->net = NULL;
	}
	
	// free the container
    Py_TYPE(self)->tp_free((PyObject*)self);
}


// EnableDebug
PyObject* PyTensorNet_EnableDebug( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
	self->net->EnableDebug();
	Py_RETURN_NONE;
}


// EnableLayerProfiler
PyObject* PyTensorNet_EnableLayerProfiler( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
	self->net->EnableLayerProfiler();
	Py_RETURN_NONE;
}


// PrintProfilerTimes
PyObject* PyTensorNet_PrintProfilerTimes( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
	self->net->PrintProfilerTimes();
	Py_RETURN_NONE;
}


// GetNetworkTime
static PyObject* PyTensorNet_GetNetworkTime( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
    return PyFloat_FromDouble(self->net->GetNetworkTime());
}


// GetNetworkFPS
static PyObject* PyTensorNet_GetNetworkFPS( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
    return PyFloat_FromDouble(self->net->GetNetworkFPS());
}


// GetModelType
static PyObject* PyTensorNet_GetModelType( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
    return Py_BuildValue("s", modelTypeToStr(self->net->GetModelType()));
}


// GetModelPath
static PyObject* PyTensorNet_GetModelPath( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
    return Py_BuildValue("s", self->net->GetModelPath());
}


// GetPrototxtPath
static PyObject* PyTensorNet_GetPrototxtPath( PyTensorNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "tensorNet invalid object instance");
		return NULL;
	}
	
    return Py_BuildValue("s", self->net->GetPrototxtPath());
}


//-------------------------------------------------------------------------------
static PyTypeObject pyTensorNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyTensorNet_Methods[] = 
{
	{ "EnableDebug", (PyCFunction)PyTensorNet_EnableDebug, METH_NOARGS, "Enable TensorRT debug messages and device synchronization"},
	{ "EnableLayerProfiler", (PyCFunction)PyTensorNet_EnableLayerProfiler, METH_NOARGS, "Enable the profiling of network layer execution times"},
	{ "PrintProfilerTimes", (PyCFunction)PyTensorNet_PrintProfilerTimes, METH_NOARGS, "Print out performance timing info"},	
	{ "GetNetworkTime", (PyCFunction)PyTensorNet_GetNetworkTime, METH_NOARGS, "Return the runtime of the network (in milliseconds)"},	
	{ "GetNetworkFPS", (PyCFunction)PyTensorNet_GetNetworkFPS, METH_NOARGS, "Return the runtime of the network (in frames per second)"},		
	{ "GetModelType", (PyCFunction)PyTensorNet_GetModelType, METH_NOARGS, "Return the type of model format (caffe, ONNX, UFF, or custom)"},
	{ "GetModelPath", (PyCFunction)PyTensorNet_GetModelPath, METH_NOARGS, "Return the path to the network model file on disk"},
	{ "GetPrototxtPath", (PyCFunction)PyTensorNet_GetPrototxtPath, METH_NOARGS, "Return the path to the network prototxt file on disk"},
	{NULL}  /* Sentinel */
};

// Get type
PyTypeObject* PyTensorNet_Type()
{
	return &pyTensorNet_Type;
}

// Register type
bool PyTensorNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	pyTensorNet_Type.tp_name 	  = PY_INFERENCE_MODULE_NAME ".tensorNet";
	pyTensorNet_Type.tp_basicsize   = sizeof(PyTensorNet_Object);
	pyTensorNet_Type.tp_flags 	  = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyTensorNet_Type.tp_methods     = pyTensorNet_Methods;
	pyTensorNet_Type.tp_new 	       = PyTensorNet_New;
	pyTensorNet_Type.tp_init	       = NULL; /*(initproc)PyTensorNet_Init;*/
	pyTensorNet_Type.tp_dealloc	  = (destructor)PyTensorNet_Dealloc;
	pyTensorNet_Type.tp_doc  	  = "TensorRT DNN Base Object";
	 
	if( PyType_Ready(&pyTensorNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "tensorNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyTensorNet_Type);
    
	if( PyModule_AddObject(module, "tensorNet", (PyObject*)&pyTensorNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "tensorNet PyModule_AddObject('tensorNet') failed\n");
		return false;
	}
	
	return true;
}
