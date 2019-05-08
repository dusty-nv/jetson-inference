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

#include <Python.h>

#if PY_MAJOR_VERSION >= 3
#define PYTHON_3
#endif

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
	printf("Jetson.Inference -- initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	PyObject* module = PyModule_Create(&pyInferenceModuleDef);
	printf("Jetson.Inference -- done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	return module;
}

#else
PyMODINIT_FUNC
initjetson_inference_python(void)
{
	printf("Jetson.Inference -- initializing Python %i.%i bindings...\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
	Py_InitModule("jetson_inference_python", pyInferenceFunctions);
	printf("Jetson.Inference -- done Python %i.%i binding initialization\n", PY_MAJOR_VERSION, PY_MINOR_VERSION);
}
#endif


