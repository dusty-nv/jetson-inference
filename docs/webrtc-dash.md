<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="webrtc-flask.md">Back</a> | <a href="webrtc-dash.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# Plotly Dash

[Plotly Dash](https://plotly.com/dash/) is a Python-based web framework for building data-driven dashboard analytics and interactive UI's.  On the frontend it uses [React.js](https://reactjs.org/) client-side, which connects state changes to Python callbacks running on the server.  With it, you can quickly develop more complex applications that integrate with backend processing pipelines.  In this example (found under [`python/www/dash`](../python/www/dash)), users can dynamically create streams, load DNN models, visualize events, and setup extendable actions triggered by events:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/webrtc-dash.jpg" width="850">

As before, it uses WebRTC for streaming live video and TensorRT for inferencing.  Note that this sample is still under development as a proof-of-concept.  For reference, the project is structured as follows:

  * [`app.py`](../python/www/dash/app.py) (webserver)
  * [`actions/`](../python/www/dash/actions) (action plugins)
  * [`assets/`](../python/www/dash/assets) (CSS/JavaScript/images)
  * [`layout/`](../python/www/dash/layout) (UI components)
  * [`server/`](../python/www/dash/server) (backend streaming/processing)

## Running the Example

Launching app.py will start the dashboard, along with a backend process that runs the WebRTC capture/transport, inferencing, and event analytics.  Running the backend in it's own process allows for multiple webserver workers to load-balance in deployment (i.e. with Gunicorn or other production WSGI webservers)

``` bash
$ cd jetson-inference/python/www/dash
$ pip3 install -r requirements.txt
$ python3 app.py --detection=ssd-mobilenet-v2 --pose=resnet18-hand --action=resnet18-kinetics
```

> **note**: it's recommended to enable [HTTPS/SSL](webrtc-server.md#enabling-https--ssl) before running this.

You should then be able to navigate your browser to `https://<JETSON-IP>:8050` and start configuring the system.  8050 is the default port used, but you can change that with the `--port=N` command-line argument.  There are also various settings that you can change through `data/config.json` (which gets written with the defaults from [`config.py`](../python/www/dash/config.py) the first time you run the app).

### Loading DNN Models

The first thing to do is to load some DNN models by going to the `Models -> Load Model` menu.  Currently it supports loading pre-trained classification and detection models, in addition to importing custom-trained ONNX models:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/webrtc-dash-model-load.jpg" width="400">

After selecting the model in the dialog, you should see a status message appear at the bottom of the main page when it's been loaded.  Typically this takes a 5-10 seconds, but if it's the first time you've loaded that particular model it could take TensorRT a few minutes to generate the network engine (to avoid this delay, it's recommended to load the model once with one of the imagenet/detectnet programs prior to running the webapp)

To load a customized classification or detection ONNX model that you trained with PyTorch from the Hello AI World tutorial (i.e. with train.py or train_ssd.py), switch to the `Import` tab:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/webrtc-dash-model-import.jpg" width="400">

It's expected that your model already exists somewhere on the server, and you can fill out the input/output layer names with the same ones like used [here](pytorch-cat-dog.md#processing-images-with-tensorrt) for classification and [here](pytorch-ssd.md#processing-images-with-tensorrt) for detection.

### Creating Streams

By opening the `Streams -> Add Stream` menu, you can specify video sources to stream, along with the DNN models that you want applied during processing:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/webrtc-dash-add-stream.jpg" width="400">

The syntax for connecting to various types of cameras and video devices can be found on the [Camera Streaming and Multimedia](aux-streaming.md) page.  Note that WebRTC input from browser webcams is not yet supported in this sample, but will be added.  After you add a stream, you can open it's video player by selecting it from the `Streams` menu.

<p align="right">Next | <b><a href="aux-streaming.md">Camera Streaming and Multimedia</a></b>
<br/>
Back | <b><a href="webrtc-flask.md">Flask + REST</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
