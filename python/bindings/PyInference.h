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

	#ifndef PYLONG_AS_LONG
	#define PYLONG_AS_LONG(x)			PyLong_AsLong(x)
	#endif

	#ifndef PYLONG_FROM_LONG
	#define PYLONG_FROM_LONG(x)			PyLong_FromLong(x)
	#endif

	#ifndef PYLONG_FROM_UNSIGNED_LONG
	#define PYLONG_FROM_UNSIGNED_LONG(x)	PyLong_FromUnsignedLong(x)
	#endif

	#ifndef PYSTRING_FROM_STRING
	#define PYSTRING_FROM_STRING			PyUnicode_FromString
	#endif

	#ifndef PYSTRING_FROM_FORMAT
	#define PYSTRING_FROM_FORMAT			PyUnicode_FromFormat
	#endif

#elif PY_MAJOR_VERSION >= 2

	// Python2 defines
	#ifndef PYTHON_2
	#define PYTHON_2
	#endif

	#ifndef PYLONG_AS_LONG
	#define PYLONG_AS_LONG(x)			PyInt_AsLong(x)
	#endif

	#ifndef PYLONG_FROM_LONG
	#define PYLONG_FROM_LONG(x)			PyInt_FromLong(x)
	#endif

	#ifndef PYLONG_FROM_UNSIGNED_LONG
	#define PYLONG_FROM_UNSIGNED_LONG(x)	PyInt_FromLong(x)
	#endif

	#ifndef PYSTRING_FROM_STRING
	#define PYSTRING_FROM_STRING			PyString_FromString
	#endif

	#ifndef PYSTRING_FROM_FORMAT
	#define PYSTRING_FROM_FORMAT			PyString_FromFormat
	#endif

#endif

#ifndef PY_RETURN_BOOL
#define PY_RETURN_BOOL(x)	if(x) Py_RETURN_TRUE; else Py_RETURN_FALSE
#endif

#endif

