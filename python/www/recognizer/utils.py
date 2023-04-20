#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import flask
import http
import time

import torch
import torch.nn



def rest_property(object, attribute, type=str, key=None):
    """
    Handle the boilerplate of getting/setting a REST JSON property.
    This function handles GET and PUT requests for different datatypes.
    
    Parameters:
        object (object) -- the object that the attribute belongs to
        attribute (str) -- the name of the attribute from the object
        type (Type) -- type of the variable (int, float, str)
        key (str) -- the key to use if this is a dict
    """
    if not hasattr(object, attribute):
        raise ValueError(f"object is missing attribute '{attribute}'")
        
    if flask.request.method == 'GET':
        value = getattr(object, attribute)
        
        if key:
            value = value[key]
            
        response = flask.jsonify(value)
        
    elif flask.request.method == 'PUT':
        value = type(flask.request.get_json())
        
        if key:
            getattr(object, attribute)[key] = value
        else:
            setattr(object, attribute, value)

        response = ('', http.HTTPStatus.OK)
        
    print(f"{flask.request.remote_addr} - - REST {flask.request.method} {flask.request.path} => {value}")
    return response
    
    
def rest_function(getter, setter=None, type=str, key=None):
    """
    Handle the boilerplate of getting/setting a REST JSON function.
    This function handles GET and PUT requests for different datatypes.
    
    Parameters:
        getter (function) -- function for getting the variable
        setter (function) -- function for setting the variable (optional)
        type (Type) -- type of the variable (int, float, str)
        key (str) -- the key to use if this is a dict
    """
    if flask.request.method == 'GET':
        value = getter()
        
        if key:
            value = value[key]
            
        response = flask.jsonify(value)
        
    elif flask.request.method == 'PUT':
        if setter is None:
            raise ValueError("missing 'set' function needed to complete PUT request")
            
        value = type(flask.request.get_json())
        
        if key:
            setter(**{key:value})
        else:
            setter(value)
            
        response = ('', http.HTTPStatus.OK)
        
    print(f"{flask.request.remote_addr} - - REST {flask.request.method} {flask.request.path} => {value}")
    return response


_alerts = []

def alert(message, level='info', category='', duration=3500):
    """
    Log an alert that shows up on the webpage
    
    Parameters:
        message (str) -- the text string to show
        level (str) -- 'error', 'success', or 'info'
        category (str) -- unique category for supressing repetitive messages
        duration (int) -- how long to show the alert (in milliseconds)
        unique (bool) -- if true, 
    """
    _alerts.append({
        'id': len(_alerts),
        'time': round(time.time()*1000), #datetime.datetime.now().strftime('%I:%M:%S'),
        'level': level,
        'category': category,
        'message': message,
        'duration': duration
    })
    
    if len(_alerts) > 25:
        _alerts.pop(0)

def alerts(since=0):
    """
    Retrieve the alerts since the given timestamp (in milliseconds)
    """
    if len(_alerts) == 0:
        return []
        
    for i in range(len(_alerts)-1, -1, -1):
        if _alerts[i]['time'] < since:
            i += 1
            break

    if i >= len(_alerts):
        return []

    return _alerts[i:]
  
def reshape_model(model, arch, num_classes):
	"""
    Reshape a model's output layers for the given number of classes
    """
	if arch.startswith("resnet"):
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
		print("=> reshaped ResNet fully-connected layer with: " + str(model.fc))

	elif arch.startswith("alexnet"):
		model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
		print("=> reshaped AlexNet classifier layer with: " + str(model.classifier[6]))

	elif arch.startswith("vgg"):
		model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)
		print("=> reshaped VGG classifier layer with: " + str(model.classifier[6]))

	elif arch.startswith("squeezenet"):
		model.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
		model.num_classes = num_classes
		print("=> reshaped SqueezeNet classifier layer with: " + str(model.classifier[1]))

	elif arch.startswith("densenet"):
		model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes) 
		print("=> reshaped DenseNet classifier layer with: " + str(model.classifier))

	elif arch.startswith("efficientnet"):
		model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
		print(f"=> reshaped {arch} classifier layer with: " + str(model.classifier[1]))
      
	elif arch.startswith("mobilenet"):
		model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
		print(f"=> reshaped {arch} classifier layer with: " + str(model.classifier[-1]))
        
	elif arch.startswith("inception"):
		model.AuxLogits.fc = torch.nn.Linear(model.AuxLogits.fc.in_features, num_classes)
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

		print("=> reshaped Inception aux-logits layer with: " + str(model.AuxLogits.fc))
		print("=> reshaped Inception fully-connected layer with: " + str(model.fc))
	
	elif arch.startswith("googlenet"):
		if model.aux_logits:
			from torchvision.models.googlenet import InceptionAux

			model.aux1 = InceptionAux(512, num_classes)
			model.aux2 = InceptionAux(528, num_classes)

			print("=> reshaped GoogleNet aux-logits layers with: ")
			print("      " + str(model.aux1))
			print("      " + str(model.aux2))
	
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
		print("=> reshaped GoogleNet fully-connected layer with:  " + str(model.fc))
        
	else:
		raise ValueError(f"classifier reshaping not supported for {arch}")

	model.num_classes = num_classes
	return model
    