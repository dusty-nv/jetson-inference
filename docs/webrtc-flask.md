<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="webrtc-html.md">Back</a> | <a href="webrtc-dash.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# Flask + REST

[Flask](https://flask.palletsprojects.com/en/2.2.x/) is a popular Python web micro-framework that routes HTTP/HTTPS requests to user Python functions and uses the [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) templating engine to generate HTML content parameterized by Python variables.  At the same time you can also easily handle backend REST requests (typically JSON), which can be used by the client to dynamically control properties and trigger content from the frontend based on user inputs.  This interactive example (found under [`python/www/flask`](../python/www/flask)) has multiple DNNs that you can toggle simulateously from the webapp and control their various settings with the UI in realtime.

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/webrtc-flask.jpg" width="600">

It also uses [Bootstrap](https://getbootstrap.com/) CSS for styling and the UI components.  The main source files for this example are as follows:

  * [`app.py`](../python/www/flask/app.py) (webserver)
  * [`stream.py`](../python/www/flask/stream.py) (WebRTC streaming thread)
  * [`model.py`](../python/www/flask/model.py) (DNN inferencing)
  * [`index.html`](../python/www/flask/templates/index.html) (frontend presentation)

## Running the Example

Launching app.py will start a Flask webserver, along with a streaming thread that runs the WebRTC capture/transport and inferencing code:

``` bash
$ cd jetson-inference/python/www/flask
$ python3 app.py --classification=resnet-18 --detection=ssd-mobilenet-v2
```

> **note**: using browser webcams requires [HTTPS/SSL](webrtc.md#enabling-https--ssl) to be enabled

You should then be able to navigate your browser to `https://<JETSON-IP>:8050` and start the stream.  8050 is the default port used, but you can change that with the `--port=N` command-line argument.  It's also configured by default for WebRTC input and output, but if you want to use a different [video input device](aux-streaming.md#input-streams), you can set that with the `--input` argument (for example, `--input=/dev/video0` for a V4L2 camera that's directly attached to your Jetson).

### Loading DNN Models

This example supports loading multiple DNN models which can run simultaneously (classification, detection, segmentation, pose estimation, action recognition, and background removal).  When you launch the app, you can pick which models are loaded like so:

``` bash
$ python3 app.py \
    --classification=resnet18 \
    --detection=ssd-mobilenet-v2 \
    --segmentation=fcn-resnet18-mhp \
    --pose=resnet18-body \
    --action=resnet18-kinetics \
    --background=u2net
```

> **note**: depending on the Jetson you are using and the other processes running, you may not have enough memory available to load all of these models at once or the compute capacity to run them all in realtime.

## REST Queries

This app takes the core HTML/JavaScript code for doing WebRTC from the [previous example](webrtc-html.md) and builds on it with REST JSON queries.  You can see the backend stubs for these in [app.py](../python/www/flask/app.py), which JavaScript queries from the client in [index.html](../python/www/flask/templates/index.html).  Templates and macros are used to reduce the amount of boilerplate code for these and makes it quick to add new settings:

```
# backend - app.py (Python)
@app.route('/classification/enabled', methods=['GET', 'PUT'])
def classification_enabled():
   return rest_property(stream.models['classification'].IsEnabled, stream.models['classification'].SetEnabled, bool)
   
@app.route('/classification/confidence_threshold', methods=['GET', 'PUT'])
def classification_confidence_threshold():
   return rest_property(stream.models['classification'].net.GetThreshold, stream.models['classification'].net.SetThreshold, float)
	   
# frontend - index.html (Jinja/HTML/JavaScript)
{{ checkbox('classification_enabled', '/classification/enabled', 'Classification Enabled') }}
{{ slider('classification_confidence_threshold', '/classification/confidence_threshold', 'Confidence Threshold') }}
```

These implement the controls for the classification model, and there are others for the different type of DNNs.  

[`rest_property()`](../python/www/flask/utils.py) is backend utility function in Python that handles `GET` and `PUT` REST requests for getting/setting user-defined attributes.  `checkbox()` and `slider` are Jinja macros that render the HTML components for the controls and JavaScript for executing the REST queries.  If you're wondering what the `{{ ... }}` code is in index.html, those are [Jinja](https://jinja.palletsprojects.com/en/3.1.x/templates/) expressions.

<p align="right">Next | <b><a href="webrtc-flask.md">Flask</a></b>
<br/>
Back | <b><a href="webrtc-server.md">WebRTC Server</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
