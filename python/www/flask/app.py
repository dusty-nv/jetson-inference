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
import flask
import argparse

from stream import Stream
from utils import rest_property
    
    
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default=os.getenv('SSL_KEY'), type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default=os.getenv('SSL_CERT'), type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--title", default='Hello AI World', type=str, help="the title of the webpage as shown in the browser")
parser.add_argument("--input", default='webrtc://@:8554/input', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")
parser.add_argument("--classification", default='', type=str, help="load classification model (see imageNet arguments)")
parser.add_argument("--detection", default='', type=str, help="load object detection model (see detectNet arguments)")
parser.add_argument("--segmentation", default='', type=str, help="load semantic segmentation model (see segNet arguments)")
parser.add_argument("--background", default='', type=str, help="load background removal model (see backgroundNet arguments)")
parser.add_argument("--action", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--pose", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--labels", default='', type=str, help="path to labels.txt for loading a custom model")
parser.add_argument("--colors", default='', type=str, help="path to colors.txt for loading a custom model")
parser.add_argument("--input-layer", default='', type=str, help="name of input layer for loading a custom model")
parser.add_argument("--output-layer", default='', type=str, help="name of output layer(s) for loading a custom model (comma-separated if multiple)")

args = parser.parse_known_args()[0]
    
    
# create Flask & stream instance
app = flask.Flask(__name__)
stream = Stream(args)

# Flask routes
@app.route('/')
def index():
    return flask.render_template('index.html', title=args.title, send_webrtc=args.input.startswith('webrtc'),
                                 input_stream=args.input, output_stream=args.output,
                                 classification=os.path.basename(args.classification), 
                                 detection=os.path.basename(args.detection), 
                                 segmentation=os.path.basename(args.segmentation), 
                                 pose=os.path.basename(args.pose),
                                 action=os.path.basename(args.action), 
                                 background=os.path.basename(args.background))

if args.classification:
    @app.route('/classification/enabled', methods=['GET', 'PUT'])
    def classification_enabled():
        return rest_property(stream.models['classification'].IsEnabled, stream.models['classification'].SetEnabled, bool)
        
    @app.route('/classification/confidence_threshold', methods=['GET', 'PUT'])
    def classification_confidence_threshold():
        return rest_property(stream.models['classification'].net.GetThreshold, stream.models['classification'].net.SetThreshold, float)
      
    @app.route('/classification/output_smoothing', methods=['GET', 'PUT'])
    def classification_output_smoothing():
        return rest_property(stream.models['classification'].net.GetSmoothing, stream.models['classification'].net.SetSmoothing, float)
        
if args.detection:
    @app.route('/detection/enabled', methods=['GET', 'PUT'])
    def detection_enabled():
        return rest_property(stream.models['detection'].IsEnabled, stream.models['detection'].SetEnabled, bool)
      
    @app.route('/detection/confidence_threshold', methods=['GET', 'PUT'])
    def detection_confidence_threshold():
        return rest_property(stream.models['detection'].net.GetConfidenceThreshold, stream.models['detection'].net.SetConfidenceThreshold, float)
      
    @app.route('/detection/clustering_threshold', methods=['GET', 'PUT'])
    def detection_clustering_threshold():
        return rest_property(stream.models['detection'].net.GetClusteringThreshold, stream.models['detection'].net.SetClusteringThreshold, float)
        
    @app.route('/detection/overlay_alpha', methods=['GET', 'PUT'])
    def detection_overlay_alpha():
        return rest_property(stream.models['detection'].net.GetOverlayAlpha, stream.models['detection'].net.SetOverlayAlpha, float)
        
    @app.route('/detection/tracking_enabled', methods=['GET', 'PUT'])
    def detection_tracking_enabled():
        return rest_property(stream.models['detection'].net.IsTrackingEnabled, stream.models['detection'].net.SetTrackingEnabled, bool)

    @app.route('/detection/tracking_min_frames', methods=['GET', 'PUT'])
    def detection_tracking_min_frames():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='minFrames')

    @app.route('/detection/tracking_drop_frames', methods=['GET', 'PUT'])
    def detection_tracking_drop_frames():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='dropFrames')

    @app.route('/detection/tracking_overlap_threshold', methods=['GET', 'PUT'])
    def detection_tracking_overlap_threshold():
        return rest_property(stream.models['detection'].net.GetTrackingParams, stream.models['detection'].net.SetTrackingParams, int, key='overlapThreshold')
   
if args.segmentation:
    @app.route('/segmentation/enabled', methods=['GET', 'PUT'])
    def segmentation_enabled():
        return rest_property(stream.models['segmentation'].IsEnabled, stream.models['segmentation'].SetEnabled, bool)
    
    @app.route('/segmentation/overlay_alpha', methods=['GET', 'PUT'])
    def segmentation_overlay_alpha():
        return rest_property(stream.models['segmentation'].net.GetOverlayAlpha, stream.models['segmentation'].net.SetOverlayAlpha, float)
  
if args.pose:
    @app.route('/pose/enabled', methods=['GET', 'PUT'])
    def pose_enabled():
        return rest_property(stream.models['pose'].IsEnabled, stream.models['pose'].SetEnabled, bool)
      
    @app.route('/pose/confidence_threshold', methods=['GET', 'PUT'])
    def pose_confidence_threshold():
        return rest_property(stream.models['pose'].net.GetThreshold, stream.models['pose'].net.SetThreshold, float)
   
if args.action:
    @app.route('/action/enabled', methods=['GET', 'PUT'])
    def action_enabled():
        return rest_property(stream.models['action'].IsEnabled, stream.models['action'].SetEnabled, bool)
        
    @app.route('/action/confidence_threshold', methods=['GET', 'PUT'])
    def action_confidence_threshold():
        return rest_property(stream.models['action'].net.GetThreshold, stream.models['action'].net.SetThreshold, float)
        
    @app.route('/action/skip_frames', methods=['GET', 'PUT'])
    def action_skip_frames():
        return rest_property(stream.models['action'].net.GetSkipFrames, stream.models['action'].net.SetSkipFrames, int)
        
if args.background:
    @app.route('/background/enabled', methods=['GET', 'PUT'])
    def background_enabled():
        return rest_property(stream.models['background'].IsEnabled, stream.models['background'].SetEnabled, bool)
        
    
# start stream thread
stream.start()

# check if HTTPS/SSL requested
ssl_context = None

if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)
    
# start the webserver
app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)