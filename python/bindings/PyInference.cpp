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

#include "PyInference.h"

#include "PyTensorNet.h"
#include "PyImageNet.h"


extern "C"
PyObject* PyInference_Test( PyObject* self, PyObject* args )
{
	printf("hello from Test!!\n");
	Py_RETURN_NONE;
}

extern "C"
PyObject* PyInference_String( PyObject* self, PyObject* args )
{
	return Py_BuildValue("s", "hello, this is a string from MyString()");
}

extern "C"
PyObject* PyInference_Print( PyObject* self, PyObject* args )
{
	const char* text = NULL;

	printf("hello from Print!!\n");

	if( !PyArg_ParseTuple(args, "s", &text) )
		return NULL;

	printf("PyInference:  %s\n", text);
	Py_RETURN_NONE;
}


static PyMethodDef pyInferenceFunctions[] =
{
	{ "MyTest", PyInference_Test, METH_VARARGS, "Test function." },
	{ "MyPrint", PyInference_Print, METH_VARARGS, "Print a string." },
	{ "MyString", PyInference_String, METH_VARARGS, "Return a test string." },
	{ NULL, NULL, 0, NULL }
};


// register object types
bool PyInference_Register( PyObject* module )
{
	printf("jetson.inference -- registering module types...\n");
	
	if( !PyTensorNet_Register(module) )
		printf("jetson.inference -- failed to register tensorNet type\n");
	
	if( !PyImageNet_Register(module) )
		printf("jetson.inference -- failed to register imageNet type\n");
	
	printf("jetson.inference -- done registering module types\n");
	return true;
}

#ifdef PYTHON_3
static struct PyModuleDef pyInferenceModuleDef = {
        PyModuleDef_HEAD_INIT,
        "jetson_inference_python",
        NULL,
        -1,
        pyInferenceFunctions
};

PyMODINIT_FUNC
PyInit_jetson_inference_python(void)
{
	printf("jetson.inference -- initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	
	// create the module
	PyObject* module = PyModule_Create(&pyInferenceModuleDef);
	
	if( !module )
	{
		printf("jetson.inference -- PyModule_Create() failed\n");
		return NULL;
	}
	
	// register types
	if( !PyInference_Register(module) )
		printf("jetson.inference -- failed to register module types\n");
	
	printf("jetson.inference -- done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	return module;
}

#else
PyMODINIT_FUNC
initjetson_inference_python(void)
{
	printf("jetson.inference -- initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	
	// create the module
	PyObject* module = Py_InitModule("jetson_inference_python", pyInferenceFunctions);
	
	if( !module )
	{
		printf("jetson.inference -- Py_InitModule() failed\n");
		return;
	}
	
	// register types
	if( !PyInference_Register(module) )
		printf("jetson.inference -- failed to register module types\n");
	
	printf("jetson.inference -- done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
}
#endif


