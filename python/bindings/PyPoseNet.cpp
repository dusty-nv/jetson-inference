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
#include "PyPoseNet.h"

#include "poseNet.h"
#include "logging.h"

#include "../../utils/python/bindings/PyCUDA.h"


//-----------------------------------------------------------------------------------------
typedef struct {
	PyObject_HEAD
	poseNet::ObjectPose::Keypoint keypoint;
	poseNet* net;
} PyPoseKeypoint_Object;

#define DOC_POSE_KEYPOINT "Object Pose Keypoint Result\n\n" \
				  "----------------------------------------------------------------------\n" \
				  "Data descriptors defined here:\n\n" \
				  "ID\n" \
				  "    Type ID of the keypoint - the name can be retrieved with poseNet.GetKeypointName()\n\n" \
				  "x\n" \
				  "    Detected x-coordinate of the keypoint\n\n" \
				  "y\n" \
				  "    Detected y-coordinate of the keypoint\n\n" 

// New
static PyObject* PyPoseKeypoint_New( PyTypeObject* type, PyObject* args, PyObject* kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyPoseKeypoint_New()\n");
	
	// allocate a new container
	PyPoseKeypoint_Object* self = (PyPoseKeypoint_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "poseNet.ObjectPose.Keypoint tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->keypoint.ID = 0;
	self->keypoint.x = 0;
	self->keypoint.y = 0;
	
	self->net = NULL;
	
	return (PyObject*)self;
}
 
// Init
static int PyPoseKeypoint_Init( PyPoseKeypoint_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyPoseKeypoint_Init()\n");
	
	// parse arguments
	int id = 0;
	
	float x = 0.0;
	float y = 0.0;

	static char* kwlist[] = {"id", "x", "y", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|iff", kwlist, &id, &x, &y))
		return -1;

	if( id < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "invalid keypoint ID was < 0");
		return -1;
	}
	
	self->keypoint.ID = id;
	self->keypoint.x = x;
	self->keypoint.y = y;
	
	return 0;
}

// Deallocate
static void PyPoseKeypoint_Dealloc( PyPoseKeypoint_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyPoseKeypoint_Dealloc()\n");

	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// ToString
static PyObject* PyPoseKeypoint_ToString( PyPoseKeypoint_Object* self )
{
	// create name string
	char name_str[256];
	
	if( self->net != NULL )
		sprintf(name_str, "(%s)", self->net->GetKeypointName(self->keypoint.ID));
	else
		memset(name_str, 0, sizeof(name_str));
		
	// format string
	char str[1024];

	sprintf(str, 
		   "<poseNet.ObjectPose.Keypoint object>\n"
		   "   -- ID:  %i %s\n"
		   "   -- x:   %g\n"
		   "   -- y:   %g\n",
		   self->keypoint.ID, name_str, self->keypoint.x, self->keypoint.y);

	return PYSTRING_FROM_STRING(str);
}

// GetID
static PyObject* PyPoseKeypoint_GetID( PyPoseKeypoint_Object* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->keypoint.ID);
}

// SetID
static int PyPoseKeypoint_SetID( PyPoseKeypoint_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Keypoint.ID attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	if( arg < 0 )
		arg = 0;

	self->keypoint.ID = arg;
	return 0;
}

// GetX
static PyObject* PyPoseKeypoint_GetX( PyPoseKeypoint_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->keypoint.x);
}

// SetX
static int PyPoseKeypoint_SetX( PyPoseKeypoint_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Keypoint.x attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->keypoint.x = arg;
	return 0;
}

// GetY
static PyObject* PyPoseKeypoint_GetY( PyPoseKeypoint_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->keypoint.y);
}

// SetY
static int PyPoseKeypoint_SetY( PyPoseKeypoint_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Keypoint.y attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->keypoint.y = arg;
	return 0;
}

  
static PyGetSetDef pyPoseKeypoint_GetSet[] = 
{
	{ "ID", (getter)PyPoseKeypoint_GetID, (setter)PyPoseKeypoint_SetID, "Type ID of the keypoint", NULL},
	{ "x", (getter)PyPoseKeypoint_GetX, (setter)PyPoseKeypoint_SetX, "x-coordinate of the keypoint", NULL},
	{ "y", (getter)PyPoseKeypoint_GetY, (setter)PyPoseKeypoint_SetY, "y-coordinate of the keypoint", NULL},
	{ NULL } /* Sentinel */
};

static PyTypeObject pyPoseKeypoint_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

//-----------------------------------------------------------------------------------------
typedef struct {
	PyObject_HEAD
	
	//poseNet::ObjectPose pose;   // causes segfault because has C++ vectors which Python
							// doesn't allocate correctly with it's C-style malloc()
	
	int ID;
	
	float left;
	float right;
	float top;
	float bottom;
	
	PyObject* keypoints;
	PyObject* links;
	
	poseNet* net;
	
} PyObjectPose_Object;


#define DOC_OBJECT_POSE "Object Pose Estimation Result\n\n" \
				  "----------------------------------------------------------------------\n" \
				  "Data descriptors defined here:\n\n" \
				  "Keypoints\n" \
				  "    List of poseNet.ObjectPose.Keypoint objects\n\n" \
				  "Links\n" \
				  "    List of (a,b) tuples, where a & b are indexes into the Keypoints list\n\n" \
				  "ID\n" \
				  "    Object ID from the image frame, starting at 0\n\n" \
				  "Left\n" \
				  "    Left bounding box coordinate\n\n" \
				  "Right\n" \
				  "    Right bounding box coordinate\n\n" \
				  "Top\n" \
				  "    Top bounding box coordinate\n\n" \
				  "Bottom\n" \
				  "    Bottom bounding box coordinate\n\n"
				  

// New
static PyObject* PyObjectPose_New( PyTypeObject* type, PyObject* args, PyObject* kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyObjectPose_New()\n");

	// allocate a new container
	PyObjectPose_Object* self = (PyObjectPose_Object*)type->tp_alloc(type, 0);
	
	if( !self )
	{
		PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "poseNet.ObjectPose tp_alloc() failed to allocate a new object");
		return NULL;
	}
	
	self->ID = 0;
	self->left = 0;
	self->right = 0;
	self->top = 0;
	self->bottom = 0;
	
	self->keypoints = NULL;
	self->links = NULL;
	self->net = NULL;
	
	return (PyObject*)self;
}

// Init
static int PyObjectPose_Init( PyObjectPose_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyObjectPose_Init()\n");

	// parse arguments
	int id = 0;
	static char* kwlist[] = {"id", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &id))
		return -1;

	if( id < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "invalid object pose ID was < 0");
		return -1;
	}
	
	self->ID = id;
	return 0;
}

// Deallocate
static void PyObjectPose_Dealloc( PyObjectPose_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyObjectPose_Dealloc()\n");

	Py_XDECREF(self->keypoints);
	Py_XDECREF(self->links);
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}

// Setup
static void PyObjectPose_Setup( PyObjectPose_Object* self, const poseNet::ObjectPose& pose, poseNet* net=NULL )
{
	self->ID = pose.ID;
	self->net = net;
	
	self->left = pose.Left;
	self->right = pose.Right;
	self->top = pose.Top;
	self->bottom = pose.Bottom;
	
	// create keypoints list
	const uint32_t numKeypoints = pose.Keypoints.size();
	self->keypoints = PyList_New(numKeypoints);

	for( uint32_t n=0; n < numKeypoints; n++ )
	{
		PyPoseKeypoint_Object* pyKeypoint = PyObject_New(PyPoseKeypoint_Object, &pyPoseKeypoint_Type);

		if( !pyKeypoint )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "failed to create a new poseNet.ObjectPose.Keypoint object");
			return;
		}
		
		pyKeypoint->keypoint = pose.Keypoints[n];
		pyKeypoint->net = net;
		
		PyList_SET_ITEM(self->keypoints, n, (PyObject*)pyKeypoint);
	}
	
	// create links list
	const uint32_t numLinks = pose.Links.size();
	self->links = PyList_New(numLinks);

	for( uint32_t n=0; n < numLinks; n++ )
	{
		PyObject* a = PYLONG_FROM_UNSIGNED_LONG(pose.Links[n][0]);
		PyObject* b = PYLONG_FROM_UNSIGNED_LONG(pose.Links[n][1]);

		PyObject* tuple = PyTuple_Pack(2, a, b);

		Py_DECREF(a);
		Py_DECREF(b);
	
		PyList_SET_ITEM(self->links, n, (PyObject*)tuple);
	}
}

// Sync the keypoints/links Python lists with their C++ counterparts
static bool PyObjectPose_Sync( PyObjectPose_Object* self, poseNet::ObjectPose& pose )
{
	// sync keypoints list
	const Py_ssize_t numKeypoints = PyList_GET_SIZE(self->keypoints);
	
	for( Py_ssize_t n=0; n < numKeypoints; n++ )
	{
		PyPoseKeypoint_Object* pyKeypoint = (PyPoseKeypoint_Object*)PyList_GET_ITEM(self->keypoints, n);
		
		if( PyObject_IsInstance((PyObject*)pyKeypoint, (PyObject*)&pyPoseKeypoint_Type) != 1 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "all objects in the Keypoints list must be of type poseNet.ObjectPose.Keypoint");
			return false;
		}
		
		pose.Keypoints.push_back(pyKeypoint->keypoint);
	}
	
	// sync links list
	const Py_ssize_t numLinks = PyList_GET_SIZE(self->links);
	
	for( Py_ssize_t n=0; n < numLinks; n++ )
	{
		PyObject* tuple = PyList_GET_ITEM(self->links, n);
		
		if( !PyTuple_Check(tuple) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "all objects in the Links list must be (int,int) tuples");
			return false;
		}

		int a = 0;
		int b = 0;
		
		if( !PyArg_ParseTuple(tuple, "ii", &a, &b) )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "failed to parse Links list - all objects in the Links list must be (int,int) tuples");
			return false;
		}
		
		if( a < 0 || b < 0 || a >= numKeypoints || b >= numKeypoints )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "Link entry had an out-of-bounds index into the Keypoints list");
			return false;
		}
		
		pose.Links.push_back({(uint32_t)a, (uint32_t)b});
	}
	
	// sync other members
	pose.ID = self->ID;
	
	pose.Left = self->left;
	pose.Right = self->right;
	pose.Top = self->top;
	pose.Bottom = self->bottom;
	
	return true;
}

// ToString
static PyObject* PyObjectPose_ToString( PyObjectPose_Object* self )
{
	// format string
	char str[1024];

	sprintf(str, 
		   "<poseNet.ObjectPose object>\n"
		   "   -- ID:        %i\n"
		   "   -- Left:      %g\n"
		   "   -- Top:       %g\n"
		   "   -- Right:     %g\n"
		   "   -- Bottom:    %g\n"
		   "   -- Keypoints: %zu\n"
		   "   -- Links:     %zu\n",
		   self->ID, self->left, self->top,
		   self->right, self->bottom,
		   PyList_GET_SIZE(self->keypoints),
		   PyList_GET_SIZE(self->links));

	// TODO append strings for keypoints/links
	return PYSTRING_FROM_STRING(str);
}

// GetID
static PyObject* PyObjectPose_GetID( PyObjectPose_Object* self, void* closure )
{
	return PYLONG_FROM_UNSIGNED_LONG(self->ID);
}

// SetID
static int PyObjectPose_SetID( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.ID attribute");
		return -1;
	}

	int arg = PYLONG_AS_LONG(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	if( arg < 0 )
		arg = 0;

	self->ID = arg;
	return 0;
}

// GetLeft
static PyObject* PyObjectPose_GetLeft( PyObjectPose_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->left);
}

// SetLeft
static int PyObjectPose_SetLeft( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Left attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->left = arg;
	return 0;
}

// GetRight
static PyObject* PyObjectPose_GetRight( PyObjectPose_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->right);
}

// SetRight
static int PyObjectPose_SetRight( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Right attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->right = arg;
	return 0;
}

// GetTop
static PyObject* PyObjectPose_GetTop( PyObjectPose_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->top);
}

// SetTop
static int PyObjectPose_SetTop( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Top attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->top = arg;
	return 0;
}

// GetBottom
static PyObject* PyObjectPose_GetBottom( PyObjectPose_Object* self, void* closure )
{
	return PyFloat_FromDouble(self->bottom);
}

// SetTop
static int PyObjectPose_SetBottom( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Bottom attribute");
		return -1;
	}

	const double arg = PyFloat_AsDouble(value);

	if( PyErr_Occurred() != NULL )
		return -1;

	self->bottom = arg;
	return 0;
}

// GetROI
static PyObject* PyObjectPose_GetROI( PyObjectPose_Object* self, void* closure )
{
	PyObject* left   = PyFloat_FromDouble(self->left);
	PyObject* top    = PyFloat_FromDouble(self->top);
	PyObject* right  = PyFloat_FromDouble(self->right);
	PyObject* bottom = PyFloat_FromDouble(self->bottom);

	PyObject* tuple = PyTuple_Pack(4, left, top, right, bottom);

	Py_DECREF(left);
	Py_DECREF(top);
	Py_DECREF(right);
	Py_DECREF(bottom);

	return tuple;
}

// GetKeypoints
static PyObject* PyObjectPose_GetKeypoints( PyObjectPose_Object* self, void* closure )
{
	Py_INCREF(self->keypoints);
	return self->keypoints;
}

// SetKeypoints
static int PyObjectPose_SetKeypoints( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Keypoints attribute");
		return -1;
	}

	if( !PyList_Check(value) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose.Keypoints must be a list of poseNet.ObjectPose.Keypoint objects");
		return -1;
	}
	
	Py_XDECREF(self->keypoints);
	Py_INCREF(value);
	
	self->keypoints = value;
	return 0;
}

// GetLinks
static PyObject* PyObjectPose_GetLinks( PyObjectPose_Object* self, void* closure )
{
	Py_INCREF(self->links);
	return self->links;
}

// SetLinks
static int PyObjectPose_SetLinks( PyObjectPose_Object* self, PyObject* value, void* closure )
{
	if( !value )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "Not permitted to delete poseNet.ObjectPose.Links attribute");
		return -1;
	}

	if( !PyList_Check(value) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose.Links must be a list of (int,int) tuples");
		return -1;
	}
	
	Py_XDECREF(self->links);
	Py_INCREF(value);
	
	self->links = value;
	return 0;
}

// FindKeypoint
static PyObject* PyObjectPose_FindKeypoint( PyObjectPose_Object* self, PyObject *args )
{
	if( !self )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose invalid object instance");
		return NULL;
	}

	// this function accepts either the keypoint ID (int) or the name (string)
	// in the string case, it will just look up the keypoint ID for you
	int id = 0;

	if( !PyArg_ParseTuple(args, "i", &id) )
	{
		PyErr_Clear();  // PyArg_ParseTuple will throw an exception
		
		if( self->net != NULL )
		{
			const char* name = NULL;

			if( !PyArg_ParseTuple(args, "s", &name) )
				return NULL;
			
			id = self->net->FindKeypointID(name);
			
			if( id < 0 )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose.FindKeypoint() could not find a keypoint by that name in the topology - check poseNet.GetNumKeypoints() / poseNet.GetKeypointName() for valid keypoint names");
				return NULL;
			}
		}
		else
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "this ObjectPose object wasn't created by a poseNet - please pass in a keypoint ID (int) instead of a name string");
			return NULL;
		}
	}
	
	// sync with C++ object
	poseNet::ObjectPose pose;
	
	if( !PyObjectPose_Sync(self, pose) )
		return NULL;

	return PYLONG_FROM_LONG(pose.FindKeypoint(id));
}

// FindLink
static PyObject* PyObjectPose_FindLink( PyObjectPose_Object* self, PyObject *args )
{
	if( !self )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose invalid object instance");
		return NULL;
	}

	// parse arguments
	int a = 0;
	int b = 0;

	if( !PyArg_ParseTuple(args, "ii", &a, &b))
	{
		if( self->net != NULL )
		{
			const char* name_a = NULL;
			const char* name_b = NULL;
			
			if( !PyArg_ParseTuple(args, "ss", &name_a, &name_b) )
				return NULL;

			a = self->net->FindKeypointID(name_a);
			b = self->net->FindKeypointID(name_b);
			
			if( a < 0 || b < 0 )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.ObjectPose.FindLink() could not find a keypoint by that name in the topology - check poseNet.GetNumKeypoints() / poseNet.GetKeypointName() for valid keypoint names");
				return NULL;
			}
		}
		else
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "this ObjectPose object wasn't created by a poseNet - please pass in a keypoint ID (int) instead of a name string");
			return NULL;
		}
	}

	// sync with C++ object
	poseNet::ObjectPose pose;
	
	if( !PyObjectPose_Sync(self, pose) )
		return NULL;
	
	return PYLONG_FROM_LONG(pose.FindLink(a,b));
}


static PyGetSetDef pyObjectPose_GetSet[] = 
{
	{ "ID", (getter)PyObjectPose_GetID, (setter)PyObjectPose_SetID, "ID of the detected object", NULL},
	{ "Keypoints", (getter)PyObjectPose_GetKeypoints, (setter)PyObjectPose_SetKeypoints, "List of detected keypoints - poseNet.ObjectPose.Keypoint objects", NULL},
	{ "Links", (getter)PyObjectPose_GetLinks, (setter)PyObjectPose_SetLinks, "List of detected links - (int,int) tuples", NULL},
	{ "Left", (getter)PyObjectPose_GetLeft, (setter)PyObjectPose_SetLeft, "Left bounding box coordinate", NULL},
	{ "Right", (getter)PyObjectPose_GetRight, (setter)PyObjectPose_SetRight, "Right bounding box coordinate", NULL},
	{ "Top", (getter)PyObjectPose_GetTop, (setter)PyObjectPose_SetTop, "Top bounding box coordinate", NULL},
	{ "Bottom", (getter)PyObjectPose_GetBottom, (setter)PyObjectPose_SetBottom, "Bottom bounding box coordinate", NULL},	
	{ "ROI", (getter)PyObjectPose_GetROI, NULL, "Tuple containing the ROI as (Left, Top, Right, Bottom)", NULL},
	{ NULL } /* Sentinel */
};

static PyMethodDef pyObjectPose_Methods[] = 
{
	{ "FindKeypoint", (PyCFunction)PyObjectPose_FindKeypoint, METH_VARARGS, "Find a keypoint index by it's ID or name, or return -1 if not found.  This returns an index into the Keypoints list."},
	{ "FindLink", (PyCFunction)PyObjectPose_FindLink, METH_VARARGS, "Find a link index by two keypoint ID's or names, or return -1 if not found.  This returns an index into the Links list."},
	{ NULL }  /* Sentinel */
};

static PyTypeObject pyObjectPose_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};


//-----------------------------------------------------------------------------------------
typedef struct {
	PyTensorNet_Object base;
	poseNet* net;
} PyPoseNet_Object;


#define DOC_POSENET   "Pose Estimation DNN - detects the poses of objects in an image\n\n" \
				  "Examples (jetson-inference/python/examples)\n" \
                      "     posenet.py\n" \
				  "__init__(...)\n" \
				  "     Loads an pose estimation model.\n\n" \
				  "     Parameters:\n" \
				  "       network (string) -- name of a built-in network to use\n" \
				  "                           see below for available options.\n\n" \
				  "       argv (strings) -- command line arguments passed to poseNet,\n" \
				  "                         see below for available options.\n\n" \
				  "       threshold (float) -- minimum detection threshold.\n" \
				  "                            default value is 0.15\n\n" \
 				  POSENET_USAGE_STRING


// Init
static int PyPoseNet_Init( PyPoseNet_Object* self, PyObject *args, PyObject *kwds )
{
	LogDebug(LOG_PY_INFERENCE "PyPoseNet_Init()\n");
	
	// parse arguments
	PyObject* argList     = NULL;
	const char* network   = "resnet18-body";
	float threshold       = POSENET_DEFAULT_THRESHOLD;

	static char* kwlist[] = {"network", "argv", "threshold", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "|sOf", kwlist, &network, &argList, &threshold))
		return -1;

	// determine whether to use argv or built-in network
	if( argList != NULL && PyList_Check(argList) && PyList_Size(argList) > 0 )
	{
		LogDebug(LOG_PY_INFERENCE "poseNet loading network using argv command line params\n");

		// parse the python list into char**
		const size_t argc = PyList_Size(argList);

		if( argc == 0 )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.__init()__ argv list was empty");
			return -1;
		}

		char** argv = (char**)malloc(sizeof(char*) * argc);

		if( !argv )
		{
			PyErr_SetString(PyExc_MemoryError, LOG_PY_INFERENCE "poseNet.__init()__ failed to malloc memory for argv list");
			return -1;
		}

		for( size_t n=0; n < argc; n++ )
		{
			PyObject* item = PyList_GetItem(argList, n);
			
			if( !PyArg_Parse(item, "s", &argv[n]) )
			{
				PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.__init()__ failed to parse argv list");
				return -1;
			}

			LogDebug(LOG_PY_INFERENCE "poseNet.__init__() argv[%zu] = '%s'\n", n, argv[n]);
		}

		// load the network using (argc, argv)
		Py_BEGIN_ALLOW_THREADS
		self->net = poseNet::Create(argc, argv);
		Py_END_ALLOW_THREADS
		
		// set the threshold
		self->net->SetThreshold(threshold);
		
		// free the arguments array
		free(argv);
	}
	else
	{
		Py_BEGIN_ALLOW_THREADS
		self->net = poseNet::Create(network, threshold);
		Py_END_ALLOW_THREADS
	}

	// confirm the network loaded
	if( !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet failed to load network");
		return -1;
	}

	self->base.net = self->net;
	return 0;
}


// Deallocate
static void PyPoseNet_Dealloc( PyPoseNet_Object* self )
{
	LogDebug(LOG_PY_INFERENCE "PyPoseNet_Dealloc()\n");

	// delete the network
	SAFE_DELETE(self->net);
	
	// free the container
	Py_TYPE(self)->tp_free((PyObject*)self);
}


#define DOC_PROCESS  "Perform pose estimation on the given image, returning object poses, and overlay the results..\n\n" \
				 "Parameters:\n" \
				 "  image   (capsule) -- CUDA memory capsule\n" \
				 "  width   (int)  -- width of the image (in pixels)\n" \
				 "  height  (int)  -- height of the image (in pixels)\n" \
				 "  overlay (str)  -- combination of box,labels,none flags (default is 'box')\n\n" \
				 "Returns:\n" \
				 "  [ObjectPoses] -- list containing the detected object poses (see poseNet.ObjectPose)"

// Detect
static PyObject* PyPoseNet_Process( PyPoseNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* capsule = NULL;

	const char* overlay   = "links,keypoints";
	static char* kwlist[] = {"image", "overlay", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "O|s", kwlist, &capsule, &overlay))
		return NULL;

	// get pointer to image data
	PyCudaImage* img = PyCUDA_GetImage(capsule);

	if( !img )
		return NULL;

	// run the pose estimation
	std::vector<poseNet::ObjectPose> poses;

	bool result = false;
	Py_BEGIN_ALLOW_THREADS
	result = self->net->Process(img->base.ptr, img->width, img->height, img->format, poses, poseNet::OverlayFlagsFromStr(overlay));
	Py_END_ALLOW_THREADS
	
	if( !result )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.Process() encountered an error processing the image");
		return NULL;
	}
	
	// create output objects
	const uint32_t numObjects = poses.size();
	PyObject* list = PyList_New(numObjects);

	for( uint32_t n=0; n < numObjects; n++ )
	{
		PyObjectPose_Object* pyPose = PyObject_New(PyObjectPose_Object, &pyObjectPose_Type);

		if( !pyPose )
		{
			PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.Process() failed to create a new poseNet.ObjectPose object");
			return NULL;
		}

		PyObjectPose_Setup(pyPose, poses[n], self->net);
		PyList_SET_ITEM(list, n, (PyObject*)pyPose);
	}

	return list;
}

#define DOC_OVERLAY "Overlay a list of object poses onto an image.\n\n" \
				 "Parameters:\n" \
				 "  input   (capsule) -- input image (CUDA memory capsule)\n" \
				 "  [ObjectPoses]  -- list containing the detected object poses (see poseNet.ObjectPose)" \
				 "  width   (int)  -- width of the image (in pixels)\n" \
				 "  height  (int)  -- height of the image (in pixels)\n" \
				 "  overlay (str)  -- combination of box,labels,none flags (default is 'box')\n\n" \
				 "  output  (capsule) -- output image (CUDA memory capsule)\n" \
				 "Returns:\n" \
				 "  None"

// Overlay
static PyObject* PyPoseNet_Overlay( PyPoseNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	// parse arguments
	PyObject* input_capsule = NULL;
	PyObject* output_capsule = NULL;
	PyObject* poses = NULL;

	const char* overlay   = "links,keypoints";
	static char* kwlist[] = {"input", "poses", "overlay", "output", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "OO|sO", kwlist, &input_capsule, &poses, &overlay, &output_capsule))
		return NULL;

	if( !output_capsule )
		output_capsule = input_capsule;

	// get pointers to image data
	PyCudaImage* input_img = PyCUDA_GetImage(input_capsule);
	PyCudaImage* output_img = PyCUDA_GetImage(output_capsule);
	
	if( !input_img || !output_img ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "failed to get CUDA image from input or output image argument(s)");
		return NULL;
	}
	
	if( input_img->width != output_img->width || input_img->height != output_img->height || input_img->format != output_img->format )
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "input/output images need to have matching dimensions and formats");
		return NULL;
	}	
	
	if ( !PyList_Check(poses) ) 
	{
		PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "object poses should be of type list");
		return NULL;
	}

	// construct vector of overlays
	std::vector<poseNet::ObjectPose> objectPoses;

	for( Py_ssize_t i=0; i < PyList_Size(poses); i++ ) 
	{
		PyObjectPose_Object* value = (PyObjectPose_Object*)PyList_GetItem(poses, i);

		if( PyObject_IsInstance((PyObject*)value, (PyObject*)&pyObjectPose_Type) != 1 ) 
		{
			PyErr_SetString(PyExc_TypeError, LOG_PY_INFERENCE "object poses list elements should be of type jetson.inference.poseNet.ObjectPose");
			return NULL;
		}
		
		// sync the keypoints/links Python lists with their C++ counterparts
		poseNet::ObjectPose pose;
		
		if( !PyObjectPose_Sync(value, pose) )  
			return NULL;
		
		objectPoses.push_back(pose);
	}

	// perform the overlay operation
	bool result = false;
	Py_BEGIN_ALLOW_THREADS
	
	result = self->net->Overlay(input_img->base.ptr, output_img->base.ptr, 
					    input_img->width, input_img->height, input_img->format, 
					    objectPoses, poseNet::OverlayFlagsFromStr(overlay));

	Py_END_ALLOW_THREADS
	
	if( !result ) 
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.Overlay() encountered an error");
		return NULL;
	}
	
	Py_RETURN_NONE;
}


#define DOC_GET_THRESHOLD  "Return the minimum detection threshold.\n\n" \
				 	  "Parameters:  (none)\n\n" \
					  "Returns:\n" \
					  "  (float) -- the threshold for detection"

// GetThreshold
static PyObject* PyPoseNet_GetThreshold( PyPoseNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}

	return PyFloat_FromDouble(self->net->GetThreshold());
}


#define DOC_SET_THRESHOLD  "Return the minimum detection threshold.\n\n" \
				 	  "Parameters:\n" \
					  "  (float) -- detection threshold\n\n" \
					  "Returns:  (none)"

// SetThreshold
PyObject* PyPoseNet_SetThreshold( PyPoseNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	float threshold = 0.0f;

	if( !PyArg_ParseTuple(args, "f", &threshold) )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.SetThreshold() failed to parse arguments");
		return NULL;
	}
		
	self->net->SetThreshold(threshold);
	Py_RETURN_NONE;
}


#define DOC_GET_NUM_KEYPOINTS "Return the number of keypoints in the model's pose topology.\n\n" \
				 	     "Parameters:  (none)\n\n" \
					     "Returns:\n" \
					     "  (int) -- number of keypoints in the model's pose topology"

// GetNumKeypoints
static PyObject* PyPoseNet_GetNumKeypoints( PyPoseNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(self->net->GetNumKeypoints());
}


#define DOC_GET_KEYPOINT_NAME "Return the keypoint name for the given keypoint ID.\n\n" \
				 	     "Parameters:\n" \
					     "  (int) -- index of the keypoint, between [0, GetNumKeypoints()]\n\n" \
					     "Returns:\n" \
					     "  (string) -- the text description of the keypoint"

// GetKeypointName
PyObject* PyPoseNet_GetKeypointName( PyPoseNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	int idx = 0;

	if( !PyArg_ParseTuple(args, "i", &idx) )
		return NULL;

	if( idx < 0 || idx >= self->net->GetNumKeypoints() )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "requested keypoint index is out of bounds");
		return NULL;
	}

	return Py_BuildValue("s", self->net->GetKeypointName(idx));
}


#define DOC_FIND_KEYPOINT_ID "Return the keypoint ID for the given keypoint name.\n\n" \
				 	     "Parameters:\n" \
					     "  (str) -- name of the keypoint\n\n" \
					     "Returns:\n" \
					     "  (int) -- the ID of the keypoint"

// FindKeypointID
PyObject* PyPoseNet_FindKeypointID( PyPoseNet_Object* self, PyObject* args )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	const char* keypointName = NULL;

	if( !PyArg_ParseTuple(args, "s", &keypointName) )
		return NULL;

	const int keypointID = self->net->FindKeypointID(keypointName);
	
	if( keypointID < 0 )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "couldn't find a keypoint by that name in the topology");
		return NULL;
	}

	return PYLONG_FROM_UNSIGNED_LONG(keypointID);
}


#define DOC_SET_KEYPOINT_ALPHA "Set the alpha blending value used during overlay visualization for one or all keypoint types\n\n" \
				 	      "Parameters:\n" \
					      "  alpha (float) -- desired alpha value, between 0.0 and 255.0\n" \
						 "  keypoint (int) -- optional index of the keypoint to set the alpha (otherwise will apply to all keypoints)\n" \
					      "Returns:  (none)"

// SetKeypointAlpha
PyObject* PyPoseNet_SetKeypointAlpha( PyPoseNet_Object* self, PyObject* args, PyObject *kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	float alpha = 0.0f;
	int keypoint = -1;
	
	static char* kwlist[] = {"alpha", "keypoint", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f|i", kwlist, &alpha) )
		return NULL;

	if( alpha < 0.0f || alpha > 255.0f )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet.SetKeypointAlpha() -- provided alpha value is out-of-range");
		return NULL;
	}

	if( keypoint >= 0 )
		self->net->SetKeypointAlpha(keypoint, alpha);
	else
		self->net->SetKeypointAlpha(alpha);

	Py_RETURN_NONE;
}


#define DOC_GET_KEYPOINT_SCALE "Get the scale used to calculate the radius of keypoints based on image dimensions.\n\n" \
				 	      "Parameters:  (none)\n\n" \
					      "Returns:\n" \
					      "  (float) -- the scale used to calculate the radius of keypoints based on image dimensions"
						
// GetKeypointScale
static PyObject* PyPoseNet_GetKeypointScale( PyPoseNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	return PyFloat_FromDouble(self->net->GetKeypointScale());
}


#define DOC_SET_KEYPOINT_SCALE "Set the scale used to calculate the radius of keypoint circles.\n" \
						 "This scale will be multiplied by the largest image dimension.\n\n" \
				 	      "Parameters:\n" \
					      "  scale (float) -- desired scaling factor\n" \
					      "Returns:  (none)"

// SetKeypointScale
PyObject* PyPoseNet_SetKeypointScale( PyPoseNet_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	float scale = 0.0f;
	static char* kwlist[] = {"scale", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f", kwlist, &scale) )
		return NULL;

	self->net->SetKeypointScale(scale);
	Py_RETURN_NONE;
}


#define DOC_GET_LINK_SCALE "Get the scale used to calculate the width of link lines based on image dimensions.\n\n" \
				 	  "Parameters:  (none)\n\n" \
					  "Returns:\n" \
					  "  (float) -- the scale used to calculate the width of link lines based on image dimensions"
						
// GetLinkScale
static PyObject* PyPoseNet_GetLinkScale( PyPoseNet_Object* self )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	return PyFloat_FromDouble(self->net->GetLinkScale());
}


#define DOC_SET_LINK_SCALE "Set the scale used to calculate the width of link lines.\n" \
					  "This scale will be multiplied by the largest image dimension.\n\n" \
				 	  "Parameters:\n" \
					  "  scale (float) -- desired scaling factor\n" \
					  "Returns:  (none)"

// SetLinkScale
PyObject* PyPoseNet_SetLinkScale( PyPoseNet_Object* self, PyObject* args, PyObject* kwds )
{
	if( !self || !self->net )
	{
		PyErr_SetString(PyExc_Exception, LOG_PY_INFERENCE "poseNet invalid object instance");
		return NULL;
	}
	
	float scale = 0.0f;
	static char* kwlist[] = {"scale", NULL};

	if( !PyArg_ParseTupleAndKeywords(args, kwds, "f", kwlist, &scale) )
		return NULL;

	self->net->SetLinkScale(scale);
	Py_RETURN_NONE;
}


#define DOC_USAGE_STRING     "Return the command line parameters accepted by __init__()\n\n" \
					    "Parameters:  (none)\n\n" \
					    "Returns:\n" \
					    "  (string) -- usage string documenting command-line options\n"

// Usage
static PyObject* PyPoseNet_Usage( PyPoseNet_Object* self )
{
	return Py_BuildValue("s", poseNet::Usage());
}

//-------------------------------------------------------------------------------
static PyTypeObject pyPoseNet_Type = 
{
    PyVarObject_HEAD_INIT(NULL, 0)
};

static PyMethodDef pyDetectNet_Methods[] = 
{
	{ "Process", (PyCFunction)PyPoseNet_Process, METH_VARARGS|METH_KEYWORDS, DOC_PROCESS},
	{ "Overlay", (PyCFunction)PyPoseNet_Overlay, METH_VARARGS|METH_KEYWORDS, DOC_OVERLAY},
	{ "GetThreshold", (PyCFunction)PyPoseNet_GetThreshold, METH_NOARGS, DOC_GET_THRESHOLD},
	{ "SetThreshold", (PyCFunction)PyPoseNet_SetThreshold, METH_VARARGS, DOC_SET_THRESHOLD},     
	{ "GetNumKeypoints", (PyCFunction)PyPoseNet_GetNumKeypoints, METH_NOARGS, DOC_GET_NUM_KEYPOINTS},
	{ "GetKeypointName", (PyCFunction)PyPoseNet_GetKeypointName, METH_VARARGS, DOC_GET_KEYPOINT_NAME},
	{ "FindKeypointID", (PyCFunction)PyPoseNet_FindKeypointID, METH_VARARGS, DOC_FIND_KEYPOINT_ID},
	{ "SetKeypointAlpha", (PyCFunction)PyPoseNet_SetKeypointAlpha, METH_VARARGS|METH_KEYWORDS, DOC_SET_KEYPOINT_ALPHA},
	{ "GetKeypointScale", (PyCFunction)PyPoseNet_GetKeypointScale, METH_NOARGS, DOC_GET_KEYPOINT_SCALE},
	{ "SetKeypointScale", (PyCFunction)PyPoseNet_SetKeypointScale, METH_VARARGS|METH_KEYWORDS, DOC_SET_KEYPOINT_SCALE},
	{ "GetLinkScale", (PyCFunction)PyPoseNet_GetLinkScale, METH_NOARGS, DOC_GET_LINK_SCALE},
	{ "SetLinkScale", (PyCFunction)PyPoseNet_SetLinkScale, METH_VARARGS|METH_KEYWORDS, DOC_SET_LINK_SCALE},
	{ "Usage", (PyCFunction)PyPoseNet_Usage, METH_NOARGS|METH_STATIC, DOC_USAGE_STRING},	
	{NULL}  /* Sentinel */
};

// Register type
bool PyPoseNet_Register( PyObject* module )
{
	if( !module )
		return false;
	
	/*
	 * register poseNet.ObjectPose.Keypoint type
	 */
	pyPoseKeypoint_Type.tp_name	   = PY_INFERENCE_MODULE_NAME ".poseNet.ObjectPose.Keypoint";
	pyPoseKeypoint_Type.tp_basicsize = sizeof(PyPoseKeypoint_Object);
	pyPoseKeypoint_Type.tp_flags	   = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyPoseKeypoint_Type.tp_base	   = NULL;
	pyPoseKeypoint_Type.tp_methods   = NULL;
	pyPoseKeypoint_Type.tp_getset    = pyPoseKeypoint_GetSet;
	pyPoseKeypoint_Type.tp_new       = PyPoseKeypoint_New;
	pyPoseKeypoint_Type.tp_init      = (initproc)PyPoseKeypoint_Init;
	pyPoseKeypoint_Type.tp_dealloc   = (destructor)PyPoseKeypoint_Dealloc;
	pyPoseKeypoint_Type.tp_str       = (reprfunc)PyPoseKeypoint_ToString;
	pyPoseKeypoint_Type.tp_repr      = (reprfunc)PyPoseKeypoint_ToString;
	pyPoseKeypoint_Type.tp_doc       = DOC_POSE_KEYPOINT;

	if( PyType_Ready(&pyPoseKeypoint_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet.ObjectPose.Keypoint PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyPoseKeypoint_Type);
	
	
	/*
	 * register poseNet.ObjectPose type
	 */
	pyObjectPose_Type.tp_name	 = PY_INFERENCE_MODULE_NAME ".poseNet.ObjectPose";
	pyObjectPose_Type.tp_basicsize = sizeof(PyObjectPose_Object);
	pyObjectPose_Type.tp_flags	 = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyObjectPose_Type.tp_base	 = NULL;
	pyObjectPose_Type.tp_methods	 = pyObjectPose_Methods;
	pyObjectPose_Type.tp_getset    = pyObjectPose_GetSet;
	pyObjectPose_Type.tp_new		 = PyObjectPose_New;
	pyObjectPose_Type.tp_init	 = (initproc)PyObjectPose_Init;
	pyObjectPose_Type.tp_dealloc	 = (destructor)PyObjectPose_Dealloc;
	pyObjectPose_Type.tp_str		 = (reprfunc)PyObjectPose_ToString;
	pyObjectPose_Type.tp_doc		 = DOC_OBJECT_POSE;

	// setup Keypoint as inner class for ObjectPose object
	pyObjectPose_Type.tp_dict = PyDict_New();

	if( !pyObjectPose_Type.tp_dict )
	{
		LogError(LOG_PY_INFERENCE "poseNet.ObjectPose failed to create new PyDict object\n");
		return false;
	}

	if( PyDict_SetItemString(pyObjectPose_Type.tp_dict, "Keypoint", (PyObject*)&pyPoseKeypoint_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet.ObjectPose failed to register poseNet.ObjectPose.Keypoint inner class\n");
		return false;
	}
	
	// complete registration of the poseNet.ObjectPose type
	if( PyType_Ready(&pyObjectPose_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet.ObjectPose PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyObjectPose_Type);


	/*
	 * register poseNet type
	 */
	pyPoseNet_Type.tp_name	   = PY_INFERENCE_MODULE_NAME ".poseNet";
	pyPoseNet_Type.tp_basicsize = sizeof(PyPoseNet_Object);
	pyPoseNet_Type.tp_flags	   = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
	pyPoseNet_Type.tp_base	   = PyTensorNet_Type();
	pyPoseNet_Type.tp_methods   = pyDetectNet_Methods;
	pyPoseNet_Type.tp_new	   = NULL; /*PyPoseNet_New;*/
	pyPoseNet_Type.tp_init	   = (initproc)PyPoseNet_Init;
	pyPoseNet_Type.tp_dealloc   = (destructor)PyPoseNet_Dealloc;
	pyPoseNet_Type.tp_doc	   = DOC_POSENET;
	 
	// setup ObjectPose as inner class for poseNet object
	pyPoseNet_Type.tp_dict = PyDict_New();

	if( !pyPoseNet_Type.tp_dict )
	{
		LogError(LOG_PY_INFERENCE "poseNet failed to create new PyDict object\n");
		return false;
	}

	if( PyDict_SetItemString(pyPoseNet_Type.tp_dict, "ObjectPose", (PyObject*)&pyObjectPose_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet failed to register poseNet.ObjectPose inner class\n");
		return false;
	}

	// complete registration of the poseNet type
	if( PyType_Ready(&pyPoseNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet PyType_Ready() failed\n");
		return false;
	}
	
	Py_INCREF(&pyPoseNet_Type);

	if( PyModule_AddObject(module, "poseNet", (PyObject*)&pyPoseNet_Type) < 0 )
	{
		LogError(LOG_PY_INFERENCE "poseNet PyModule_AddObject('poseNet') failed\n");
		return false;
	}
	
	return true;
}

