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

from stream import Stream


app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html', title=args.title, send_webrtc=args.input.startswith('webrtc'))
    
@app.route('/api/confidence-threshold', methods=['GET', 'PUT'])
def confidence_threshold():
    if flask.request.method == 'PUT':
        threshold = float(flask.request.get_json())
        print(f'set_confidence_threshold({threshold})')
        stream.net.SetConfidenceThreshold(threshold)
        return '', http.HTTPStatus.OK
    else:
        return flask.jsonify(stream.net.GetConfidenceThreshold())
  
@app.route('/api/clustering-threshold', methods=['GET', 'PUT'])
def clustering_threshold():
    if flask.request.method == 'PUT':
        threshold = float(flask.request.get_json())
        print(f'set_clustering_threshold({threshold})')
        stream.net.SetClusteringThreshold(threshold)
        return '', http.HTTPStatus.OK
    else:
        return flask.jsonify(stream.net.GetClusteringThreshold())
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
    parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
    parser.add_argument("--ssl-key", default=None, type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
    parser.add_argument("--ssl-cert", default=None, type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
    parser.add_argument("--title", default='Hello AI World', type=str, help="the title of the webpage as shown in the browser")
    parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
    parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")

    args = parser.parse_known_args()[0]
    
    # start stream thread
    stream = Stream(args)
    stream.start()
    
    # check if HTTPS/SSL requested
    ssl_context = None
    
    if args.ssl_cert and args.ssl_key:
        ssl_context = (args.ssl_cert, args.ssl_key)
        
    # start the webserver
    app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)