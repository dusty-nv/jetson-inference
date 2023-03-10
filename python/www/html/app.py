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

import os
import sys
import ssl
import argparse

from http.server import HTTPServer, SimpleHTTPRequestHandler
from stream import Stream

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default=os.getenv('SSL_KEY'), type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default=os.getenv('SSL_CERT'), type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")
parser.add_argument("--no-stream", action="store_true", help="disable creation of the input/output stream (serve website only)")
parser.add_argument("--classification", action="store_true", help="load classification model (see imageNet arguments)")
parser.add_argument("--detection", action="store_true", help="load object detection model (see detectNet arguments)")
parser.add_argument("--segmentation", action="store_true", help="load semantic segmentation model (see segNet arguments)")
parser.add_argument("--background", action="store_true", help="load background removal model (see backgroundNet arguments)")
parser.add_argument("--action", action="store_true", help="load action recognition model (see actionNet arguments)")
parser.add_argument("--pose", action="store_true", help="load action recognition model (see actionNet arguments)")

args = parser.parse_known_args()[0]

# start stream thread
if not args.no_stream:
    stream = Stream(args)
    stream.start()

# patch to serve javascript
SimpleHTTPRequestHandler.extensions_map['.js'] = 'text/javascript'

# start webserver
httpd = HTTPServer((args.host, args.port), SimpleHTTPRequestHandler)

if args.ssl_key and args.ssl_cert:
    httpd.socket = ssl.wrap_socket(httpd.socket, keyfile=args.ssl_key, certfile=args.ssl_cert, server_side=True)
    print('HTTPS enabled')
    
httpd.serve_forever()