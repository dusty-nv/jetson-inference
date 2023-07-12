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
 
#ifndef __PYTHON_BINDINGS_INFERENCE__
#define __PYTHON_BINDINGS_INFERENCE__

#include <Python.h>

// user-facing module name 
#define PY_INFERENCE_MODULE_NAME "jetson.inference"

// logging prefix
#define LOG_PY_INFERENCE PY_INFERENCE_MODULE_NAME " -- "

// check Python version
#if PY_MAJOR_VERSION >= 3

	// Python3 defines
	#ifndef PYTHON_3
	#define PYTHON_3
	#endif

	#define PYLONG_AS_LONG(x)				PyLong_AsLong(x)
	#define PYLONG_FROM_LONG(x)				PyLong_FromLong(x)
	#define PYLONG_FROM_UNSIGNED_LONG(x)		PyLong_FromUnsignedLong(x)
	#define PYLONG_FROM_UNSIGNED_LONG_LONG(x) 	PyLong_FromUnsignedLongLong(x)

	#define PYSTRING_CHECK					PyUnicode_Check
	#define PYSTRING_AS_STRING				PyUnicode_AsUTF8
	#define PYSTRING_FROM_STRING				PyUnicode_FromString
	#define PYSTRING_FROM_FORMAT				PyUnicode_FromFormat

#elif PY_MAJOR_VERSION >= 2

	// Python2 defines
	#ifndef PYTHON_2
	#define PYTHON_2
	#endif

	#define PYLONG_AS_LONG(x)				PyInt_AsLong(x)
	#define PYLONG_FROM_LONG(x)				PyInt_FromLong(x)
	#define PYLONG_FROM_UNSIGNED_LONG(x)		PyInt_FromLong(x)
	#define PYLONG_FROM_UNSIGNED_LONG_LONG(x)	PyInt_FromLong((long)x)

	#define PYSTRING_CHECK					PyString_Check
	#define PYSTRING_AS_STRING				PyString_AsString
	#define PYSTRING_FROM_STRING				PyString_FromString
	#define PYSTRING_FROM_FORMAT				PyString_FromFormat
	
#endif

#ifndef PY_RETURN_BOOL
#define PY_RETURN_BOOL(x)	if(x) Py_RETURN_TRUE; else Py_RETURN_FALSE
#endif

#define PYDICT_SET_ITEM(dict, key, val) { \
	PyObject* object = val; \
	if( object != NULL ) { \
		PyDict_SetItemString(dict, key, object); \
		Py_DECREF(object); \
	} \
}
	
#define PYDICT_SET_STRING(dict, key, str) { \
	const char* string = str; \
	if( string != NULL && strlen(string) > 0 ) { \
		PyObject* obj = PYSTRING_FROM_STRING(string); \
		PyDict_SetItemString(dict, key, obj); \
		Py_DECREF(obj); \
	} \
}

#define PYDICT_SET_STDSTR(dict, key, str) \
	if( str.length() > 0 ) { \
		PyObject* obj = PYSTRING_FROM_STRING(str.c_str()); \
		PyDict_SetItemString(dict, key, obj); \
		Py_DECREF(obj); \
	}
	
#define PYDICT_SET_BOOL(dict, key, val) { \
	const bool value = val; \
	PyDict_SetItemString(dict, key, value ? Py_True : Py_False); \
}

#define PYDICT_SET_INT(dict, key, val)		PYDICT_SET_ITEM(dict, key, PYLONG_FROM_LONG(val))
#define PYDICT_SET_UINT(dict, key, val)		PYDICT_SET_ITEM(dict, key, PYLONG_FROM_UNSIGNED_LONG(val))
#define PYDICT_SET_FLOAT(dict, key, val)	PYDICT_SET_ITEM(dict, key, PyFloat_FromDouble(val))

#define PYDICT_GET_INT(dict, key, output) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL ) { \
		const int value = PYLONG_AS_LONG(obj); \
		if( !PyErr_Occurred() ) \
			output = value; \
	} \
}

#define PYDICT_GET_UINT(dict, key, output) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL ) { \
		const int value = PYLONG_AS_LONG(obj); \
		if( !PyErr_Occurred() && value >= 0 ) \
			output = value; \
	} \
}

#define PYDICT_GET_FLOAT(dict, key, output) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL ) { \
		const float value = PyFloat_AsDouble(obj); \
		if( !PyErr_Occurred() ) \
			output = value; \
	} \
}

#define PYDICT_GET_BOOL(dict, key, output) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL && PyBool_Check(obj) ) { \
		output = (obj == Py_True) ? true : false; \
	} \
}

#define PYDICT_GET_STRING(dict, key, output) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL && PYSTRING_CHECK(obj) ) { \
		const char* value = PYSTRING_AS_STRING(obj); \
		if( value != NULL ) \
			output = value; \
	} \
}

#define PYDICT_GET_ENUM(dict, key, output, parser) { \
	PyObject* obj = PyDict_GetItemString(dict, key); \
	if( obj != NULL && PYSTRING_CHECK(obj) ) { \
		const char* value = PYSTRING_AS_STRING(obj); \
		if( value != NULL ) \
			output = parser(value); \
	} \
}

#endif

