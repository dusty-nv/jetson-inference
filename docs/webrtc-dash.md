<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="webrtc-flask.md">Back</a> | <a href="webrtc-recognizer.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# Plotly Dashboard

[Plotly Dash](https://plotly.com/dash/) is a Python-based web framework for building data-driven dashboards and interactive UI's.  On the frontend it uses [React.js](https://reactjs.org/) client-side, which connects state changes to Python callbacks running on the server.  With it, you can quickly develop rich visualizations that integrate with backend processing pipelines and data analytics.  In this example (found under [`python/www/dash`](../python/www/dash)), users can dynamically create streams, load DNN models, visualize events, and setup extendable actions triggered by events:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash.jpg" width="1000">

As before, it uses WebRTC for streaming live video and TensorRT for inferencing.  Note that this sample is still under development as a proof-of-concept.  For reference, the project is structured as follows:

  * [`app.py`](../python/www/dash/app.py) (webserver)
  * [`actions/`](../python/www/dash/actions) (action plugins)
  * [`assets/`](../python/www/dash/assets) (CSS/JavaScript/images)
  * [`layout/`](../python/www/dash/layout) (UI components)
  * [`server/`](../python/www/dash/server) (backend streaming/processing)

## Running the Example

Launching app.py will start the dashboard, along with a backend process that runs the WebRTC capture/transport, inferencing, and event/action triggers.  Whereas previous examples ran the streaming in a thread inside same process as the webserver, running the streaming backend in it's own independent process allows for multiple webserver workers to load-balance in deployment (i.e. with [Gunicorn](https://gunicorn.org/) or other production WSGI webservers).  These processes share metadata over REST JSON messages (with the video data residing within the streaming process).

``` bash
$ cd jetson-inference/python/www/dash
$ pip3 install -r requirements.txt
$ python3 app.py --detection=ssd-mobilenet-v2 --pose=resnet18-hand --action=resnet18-kinetics
```

> **note**: it's recommended to enable [HTTPS/SSL](webrtc-server.md#enabling-https--ssl) when running the server

You should then be able to navigate your browser to `https://<JETSON-IP>:8050` and start configuring the system.  8050 is the default port used, but you can change that with the `--port=N` command-line argument.  There are also various settings that you can change through `data/config.json` (which gets written with the defaults from [`config.py`](../python/www/dash/config.py) the first time you run the app).

### Loading DNN Models

The first thing to do is to load some DNN models by going to the `Models -> Load Model` menu.  Currently it supports loading pre-trained classification and detection models, in addition to importing custom-trained ONNX models:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-model-load.jpg" width="400">

After selecting the model in the dialog, you should see a status message appear at the bottom of the main page when it's been loaded.  Typically this takes a 5-10 seconds, but if it's the first time you've loaded that particular model it could take TensorRT a few minutes to generate the network engine (to avoid this delay, it's recommended to load the model once with one of the imagenet.py/detectnet.py programs prior to running the webapp)

#### Importing Models 

To load a customized classification or detection ONNX model that you trained with PyTorch from the Hello AI World tutorial (i.e. with [`train.py`](pytorch-cat-dog.md#re-training-resnet-18-model) or [`train_ssd.py`](https://github.com/dusty-nv/jetson-inference/blob/dev/docs/pytorch-collect-detection.md#training-your-model)), switch to the `Import` tab:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-model-import.jpg" width="400">

It's expected that your model already exists somewhere on the server, and you can fill out the input/output layer names with the same ones you would use with imagenet.py/detectnet.py (like used [here](pytorch-cat-dog.md#processing-images-with-tensorrt) for classification and [here](pytorch-ssd.md#processing-images-with-tensorrt) for detection)

### Creating Streams

By opening the `Streams -> Add Stream` menu, you can specify video sources to stream, along with the DNN models that you want applied:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-add-stream.jpg" width="400">

The syntax for connecting to various types of cameras and video devices can be found on the [Camera Streaming and Multimedia](aux-streaming.md) page.  Note that WebRTC input from browser webcams is not yet supported in this sample, but will be added.  After you add a stream, you can open it's video player by selecting it from the `Streams` menu.  The panel widgets are draggable, resizeable, and collapsable.

## Events

When the output of a DNN changes (i.e. the classification result changes, or a new object is detected) it logs an event in the system.  These events can be monitored in realtime by opening the Events Table from under the `Events` menu:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-event-table.jpg" width="750">

You can filter and sort by column in the table, and visualize the results in the Event Timeline:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-event-timeline.jpg" width="750">

The y-axis of this plot shows confidence score, the x-axis shows time, and each object class gets a different trace in the chart.  Quickly creating different types of dynamic [graphs](https://plotly.com/python/) and [tables](https://dash.plotly.com/datatable) is a strong feature of Plotly Dash, and you can extend these when creating your own apps.

## Actions

Actions are plugins that filter events and trigger user-defined code (such as alerts/notifications, playing sounds, or generating a physical response) when such an event occurs.  An action's properties are exposed through the web UI so that they are configurable at runtime:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-dash-actions.jpg" width="400">

You can add your own action types under the project's [`actions/`](../python/www/dash/actions) directory, and they'll automatically be loaded by the app at start-up and selectable from the UI.  Multiple instances of a type of action can be created by the user, each with independent settings they can control. 

For example, here's the template for an action that simply logs messages to the server's terminal:

``` python
from server import Action

class MyAction(Action):
    def __init__(self):
        super().__init__()
	   
    def on_event(self, event):
        if event.label == 'person' and event.score > 0.5:
            print("Detected a person!")  # do something        
```


Action plugins should implement the `on_event()` callback, which receives all new and updated events from the system.  The plugins then filter the events by domain-specific critera before triggering a response of some kind.  See the [`Event`](../python/www/dash/server/event.py) class for the event attributes that can be accessed.  This code all runs in the backend streaming process, and has access to the low-level data streams without impacting performance.

### Filters

Actions can implement custom event filtering logic, and/or inherit from the [`EventFilter`](../python/www/dash/server/filter.py) mix-in that implements some default filtering (like class labels, minimum confidence score, minimum number of frames, ect) which can be called with the `filter()` function.  The [`BrowserAlert`](../python/www/dash/actions/alert.py) plugin uses that below.  It then checks/sets an attribute (`alert_triggered`) which prevents the same event from triggering multiple alerts:  

``` python
from server import Server, Action, EventFilter

class BrowserAlert(Action, EventFilter):
    """
    Action that triggers browser alerts and supports event filtering.
    """
    def __init__(self):
        super(BrowserAlert, self).__init__()

    def on_event(self, event):
        if self.filter(event) and not hasattr(event, 'alert_triggered'):
            Server.alert(f"Detected '{event.label}' ({event.maxScore * 100:.1f}%)")
            event.alert_triggered = True
```

It's possible that in some more advanced scenarios, you may want to re-trigger the action when other aspects of the event changes (like a significant deviation in confidence score, or exceeding an amount of time being detected, for example)

### Properties

Plugins that have `@property` decorators will have those properties automatically exposed to the UI so that they can be dynamically modified by the user at runtime.  The client/server communication happens transparently using REST JSON queries.  For example, from [`EventFilter`](../python/www/dash/server/filter.py)

``` python
@property
def labels(self) -> str:
   return ';'.join(self._labels)
   
@labels.setter
def labels(self, labels):
   self._labels = [label.strip() for label in labels.split(';')]

@property
def min_frames(self) -> int:
   return self._min_frames
   
@min_frames.setter
def min_frames(self, min_frames):
   self._min_frames = int(min_frames)
  
@property
def min_score(self) -> float:
   return self._min_score
   
@min_frames.setter
def min_score(self, min_score):
   self._min_score = float(min_score)
```

Note the Python type hints that are specified on the getter functions - these inform the frontend of what kind of UI control to use (e.g. textbox, slider, checkbox, ect).  Supported types are `str`, `int`, `float`, and `bool`.  If the type hint is omitted, it will be assumed to be a string with a textbox input, and the user's plugin will be responsible for parsing/converting it to the desired type.

<p align="right">Next | <b><a href="webrtc-recognizer.md">Recognizer (Interactive Training)</a></b>
<br/>
Back | <b><a href="webrtc-flask.md">Flask + REST</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
