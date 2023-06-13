<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="webrtc-server.md">Back</a> | <a href="webrtc-flask.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# HTML / JavaScript

Included in this repo are various example webapps using WebRTC that are found under [`jetson-inference/python/www`](../python/www), such as:

```
+ python/
  + www/
    - dash       # Plotly Dashboard
    - html       # core HTML/JavaScript
    - flask      # Flask + REST
    - recognizer # interactive training
```

Each of these demonstrate WebRTC integration with different Python-based webserver frameworks for building out your own AI-powered interactive webapps.  These generally have similar components like the following:

```
- app.py      # server-side Python code for running the webserver
- stream.py   # server-side Python code for WebRTC streaming/inferencing
- webrtc.js   # client-side WebRTC JavaScript code
- index.html  # client-side HTML code
```

This first example is the simplest and highlights the core HTML/JavaScript code needed to playback/send WebRTC streams and apply DNN inferencing.  You can apply this to any web framework of choice should you already have a preferred frontend to integrate with.

## Running the Example

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-html.jpg" width="600">

Launching app.py will start a built-in Python webserver (which is easy to use, but isn't intended for production and can be easily changed out) along with an independent streaming thread that runs the WebRTC capture/transport and inferencing code:

``` bash
$ cd jetson-inference/python/www/html
$ python3 app.py --classification  # see below for other DNN options
```

> **note**: receiving browser webcams requires [HTTPS/SSL](webrtc-server.md#enabling-https--ssl) to be enabled

You should then be able to navigate your browser to `https://<JETSON-IP>:8050` and start the stream.  8050 is the default port used by these webapp examples, but you can change that with the `--port=N` command-line argument.  It's also configured by default for WebRTC input and output, but if you want to use a different [video input device](aux-streaming.md#input-streams), you can set that with the `--input` argument (for example, `--input=/dev/video0` for a V4L2 camera that's directly attached to your Jetson).

### Loading DNN Models

This example supports running one DNN model at a time - either classification, detection, segmentation, pose estimation, action recognition, or background removal (subsequent examples support running multiple models simulateously).  You can change the DNN when you launch the app like so:

``` bash
$ python3 app.py --classification --model=resnet18
$ python3 app.py --detection --model=ssd-mobilenet-v2
$ python3 app.py --segmentation --model=fcn-resnet18-mhp
$ python3 app.py --pose --model=resnet18-body
$ python3 app.py --action --model=resnet18-kinetics
$ python3 app.py --background --model=u2net
```

Omitting the optional `--model` argument will load the default model for that network, or if you have your own custom classification or detection model from the tutorial that you trained in PyTorch and exported to ONNX, you can use the extended command-line arguments to load it (like used [here](pytorch-cat-dog.md#processing-images-with-tensorrt) for classification and [here](pytorch-ssd.md#processing-images-with-tensorrt) for detection).

## HTML Elements

Consulting the source of [`index.html`](../python/www/html/index.html), let's walkthrough the most important steps of building your own webpages that use WebRTC:

1.  JavaScript Imports

``` html
<script type='text/javascript' src='https://webrtc.github.io/adapter/adapter-latest.js'></script>
<script type='text/javascript' src='/webrtc.js'></script>
```

2.  HTML Video Player

This should go in the page `<body>` to create the video player element:

``` html
<video id="video-player" autoplay controls playsinline muted></video>
```

3.  Start Playback

``` javascript
// playStream() is a helper function from webrtc.js that connects the specified WebRTC stream to the video player
// getWebSocketURL() is a helper function that makes a URL path of the form:  wss://<SERVER-IP>:8554/output
playStream(getWebsocketURL('output'), document.getElementById('video-player'));
```

Normally this JavaScript function would be called in `window.onload()` or from an event handler like a button's `onclick()` event (like shown in this example).  And although it's not called out above, there's also code included for enumerating a browser's webcams and sending those over WebRTC to the Jetson as a video input.  You can essentially copy & paste this code (along with `webrtc.js`) into any project to enable WebRTC.


<p align="right">Next | <b><a href="webrtc-flask.md">Flask + REST</a></b>
<br/>
Back | <b><a href="webrtc-server.md">WebRTC Server</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
