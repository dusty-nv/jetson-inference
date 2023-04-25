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
import http
import flask
import logging
import werkzeug
import argparse

from stream import Stream
from utils import rest_property, rest_function, alerts
    
    
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default=os.getenv('SSL_KEY'), type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default=os.getenv('SSL_CERT'), type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--title", default='Hello AI World | Recognizer', type=str, help="the title of the webpage as shown in the browser")
parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")

parser.add_argument("--data", default='data', type=str, help="path to store dataset and models under")
parser.add_argument("--network", "--net", default='resnet18', type=str, help="the type of DNN architecture to use (default: resnet18)")
parser.add_argument('--net-width', default=224, type=int, metavar='N', help="the input width (in pixels) of the DNN model (default: 224)")
parser.add_argument('--net-height', default=224, type=int, metavar='N', help="the input height (in pixels) of the DNN model (default: 224)")
parser.add_argument('--batch-size', default=1, type=int, metavar='N', help="training batch size to use (default: 1)")
parser.add_argument("--workers", default=2, type=int, metavar='N', help="number of training data loading workers (default: 2)")
parser.add_argument("--optimizer", default='adam', type=str, choices=['adam', 'sgd'], help="training optimizer to use (default: adam)")
parser.add_argument('--learning-rate', default=0.001, type=float, metavar='LR', help="initial training learning rate (default: 0.001)")     
parser.add_argument('--no-augmentation', action='store_false', dest='augmentation', help="disable training data image augmentation")
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help="print training progress info every N steps")

args = parser.parse_known_args()[0]
    
    
# create Flask & stream instance
app = flask.Flask(__name__)
stream = Stream(args)

# Flask routes
@app.route('/')
def index():
    return flask.render_template('index.html', title=args.title, send_webrtc=args.input.startswith('webrtc'),
                                 input_stream=args.input, output_stream=args.output,
                                 classification=os.path.basename(stream.model.onnx_path))

@app.route('/alerts', methods=['GET'])
def get_alerts():
    return alerts(flask.request.args.get('since', 0, type=int))
    
@app.route('/dataset/classes', methods=['GET'])
def dataset_classes():
    return stream.dataset.classes
    
@app.route('/dataset/active_tags', methods=['GET', 'PUT'])
def dataset_active_tags():
    return rest_function(stream.dataset.GetActiveTags, stream.dataset.SetActiveTags, str) 
   
@app.route('/dataset/recording', methods=['GET', 'PUT'])
def dataset_recording():
    return rest_property(stream.dataset, 'recording', bool)
    
@app.route('/dataset/upload', methods=['POST'])
def dataset_upload():
    file = flask.request.files.get('file')
    
    if not file or not file.filename:
        print('/dataset/upload -- invalid request (missing file)')
        return ('', http.HTTPStatus.BAD_REQUEST)

    file.filename = werkzeug.utils.secure_filename(file.filename)
    saved_path = stream.dataset.Upload(file)
    
    if not saved_path:
        print(f"/dataset/upload -- failed to save '{file.mimetype}' to dataset ({file.filename})")
        return ('', http.HTTPStatus.INTERNAL_SERVER_ERROR)
        
    return (saved_path, http.HTTPStatus.OK)
    
@app.route('/training/enabled', methods=['GET', 'PUT'])
def training_enabled():
    return rest_property(stream.model, 'training_enabled', bool)
    
@app.route('/training/stats', methods=['GET'])
def training_stats():
    return stream.model.training_stats
    
@app.route('/classification/enabled', methods=['GET', 'PUT'])
def classification_enabled():
    return rest_property(stream.model, 'inference_enabled', bool)
        
@app.route('/classification/confidence_threshold', methods=['GET', 'PUT'])
def classification_confidence_threshold():
    return rest_property(stream.model, 'classification_threshold', float)
      
@app.route('/classification/output_smoothing', methods=['GET', 'PUT'])
def classification_output_smoothing():
    return rest_property(stream.model, 'classification_smoothing', float)


# start stream thread
stream.start()

# check if HTTPS/SSL requested
ssl_context = None

if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)
    
# disable request logging (https://stackoverflow.com/a/18379764)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# start the webserver
app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)
