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

def rest_property(getter, setter, type, key=None):
    """
    Handle the boilerplate of getting/setting a REST JSON property.
    This function handles GET and PUT requests for different datatypes.
    """
    if flask.request.method == 'GET':
        value = getter()
        
        if key:
            value = value[key]
            
        response = flask.jsonify(value)
    elif flask.request.method == 'PUT':
        value = type(flask.request.get_json())
        
        if key:
            setter(**{key:value})
        else:
            setter(value)
            
        response = ('', http.HTTPStatus.OK)
        
    print(f"{flask.request.remote_addr} - - REST {flask.request.method} {flask.request.path} => {value}")
    return response
        