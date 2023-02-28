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

import http
import flask
import requests
import argparse


app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', title='Hello AI World')
    
@app.route('/api/example', methods=['GET'])
def get_example():
    print('get_example()')
    return {
        'id': 'example',
        'data': [0,1,2,3]
    }

@app.route('/api/example', methods=['POST'])
def add_example():
    msg = flask.request.get_json()
    
    print('add_example()')
    print(msg)
    
    return {'id': 'example', 'message': 'ok'}
    
@app.route('/api/example', methods=['PUT'])
def set_example():
    msg = flask.request.get_json()
    
    print('set_example()')
    print(msg)
    
    return '', http.HTTPStatus.OK

confidence_threshold = 0.75
    
@app.route('/api/confidence-threshold', methods=['GET'])
def get_confidence_threshold():
    print('get_confidence_threshold()')
    return flask.jsonify(confidence_threshold)
    
@app.route('/api/confidence-threshold', methods=['PUT'])
def set_confidence_threshold():
    global confidence_threshold
    msg = flask.request.get_json()
    
    print('set_confidence_threshold()')
    print(msg)
    print(type(msg))
    confidence_threshold = float(msg)
    
    return '', http.HTTPStatus.OK
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
    parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
    parser.add_argument("--ssl-key", default=None, type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
    parser.add_argument("--ssl-cert", default=None, type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
    
    args = parser.parse_args()
    
    # check if HTTPS/SSL requested
    ssl_context = None
    
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        
    # start the webserver
    app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True)