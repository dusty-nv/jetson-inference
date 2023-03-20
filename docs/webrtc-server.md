<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-collect-detection.md">Back</a> | <a href="webrtc-html.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# WebRTC Server

jetson-inference includes an integrated WebRTC server for streaming low-latency live video to/from web browsers that can be used for building dynamic web applications and remote data visualization tools powered by Jetson and edge AI on the backend.  WebRTC works seamlessly with DNN inferencing pipelines via the [videoSource/videoOutput](aux-streaming.md#source-code) interfaces from jetson-utils, which uses hardware-accelerated video encoding and decoding underneath through GStreamer.  It supports sending and receiving multiple streams to/from multiple clients simultaneously, and includes a built-in webserver for viewing video streams remotely without needing to build your own frontend:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-builtin.jpg" width="600">

In this screenshot of full-duplex mode, the webcam from a laptop is being streamed to a Jetson over WebRTC, where the Jetson decodes it and performs object detection using detectNet, before re-encoding the output and sending it back to the browser again via WebRTC for playback.  The round-trip latency goes largely unnoticed from an interactivity standpoint over local wireless networks.  On the client side, it's been tested with multiple browsers including Chrome/Chromium, mobile Android, and mobile iOS (Safari) using H.264 compression.

Any application using videoSource/videoOutput (including the C++ & Python examples from this repo like imagenet/imagenet.py, detectnet/detectnet.py, ect) can easily enable this WebRTC server by launching them with a stream URL of `webrtc://@:8554/my_stream` or similar.  Further examples are provided that build on these components and implement customizable frontends with more complex processing pipeines and web UI's with interactive controls.

## Enabling HTTPS / SSL

It's recommended to use secure HTTPS and SSL/TLS for transporting WebRTC streams and serving webpages so that they are encrypted.  Also, browsers require HTTPS to use a client's webcam from a PC.  To enable HTTPS, first you need to generate a self-signed SSL certificate and key:

``` bash
$ cd /jetson-inference/data
$ openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes -subj '/CN=localhost'
$ export SSL_KEY=/jetson-inference/data/key.pem
$ export SSL_CERT=/jetson-inference/data/cert.pem
```

When you set the `$SSL_KEY` and `$SSL_CERT` environment variables, they'll automatically be picked up by applications from this repo, and HTTPS will be enabled.  Otherwise, you will need to set the `--ssl-key` and `--ssl-cert` command-line arguments when you start them.

You can also store these certs anywhere you want in leui of `/jetson-inference/data` - although if you're using the Docker container, that path is recommended because it gets mounted to the host and hence your certificate will be retained after exiting the container.

When you first navigate your browser to a page that uses these self-signed certificates, it will issue you a warning since they don't originate from a trusted authority:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-ssl-warning.jpg" width="400">

You can choose to override this, and it won't re-appear again until you change certificates or your device's hostname/IP changes.

## Sending WebRTC Streams

To send any [supported video stream](aux-streaming.md#input-streams) to browsers over WebRTC for playback, simply set the videoOutput URL to `webrtc://<INTERFACE>:<PORT>/<STREAM-NAME>`.  Specifying `0.0.0.0` or `@` as the interface will bind to all network interfaces on your device.  The port used is typically 8554, but you can change it as you wish.  The stream name should be unique for each video source (you can call them whatever you want, but they shouldn't contain slashes), as these unique names allow for multiple streams to be routed by the same WebRTC server instance simultaneously and are used as the path to websockets.

``` bash
$ video-viewer /dev/video0 webrtc://@:8554/output  # stream V4L2 camera to browsers via WebRTC
$ detectnet.py csi://0 webrtc://@:8554/output      # stream MIPI CSI camera, with object detection
```

You should then be able to navigate your browser to `https://<JETSON-IP>:8554` and view the video stream like in the image above.  Various connection statistics are dynamically updated on the page, like the bitrate and number of frames and packets received/dropped.  Sometimes it can be helpful to open the browser's debug console log (`Ctrl+Shift+I` in Chrome) as status messages about the state of the WebRTC connection get printed out there.

### videoOutput Code

If you're using the videoOutput interface in your program and want to hardcode it for streaming WebRTC (as opposed to parsing the command-line), you can create it like so:


``` python
# Python
output = jetson_utils.videoOutput("webrtc://@:8554/output")

# C++
videoOutput* output = videoOutput::Create("webrtc://@:8554/output");
```

You can then use the videoOutput interface in your main loop to render frames just like you would have before like in these [examples](aux-streaming.md#source-code).


## Receiving WebRTC Streams

You can also receive streams from browser webcams via WebRTC.  To do that, enable HTTPS and set the videoInput URL similarly to above:

``` bash
$ video-viewer webrtc://@:8554/input my_video.mp4  # save browser webcam to MP4 file
$ imagenet.py webrtc://@:8554/input my_video.mp4   # save browser webcam to MP4 file (applying classification)
```

> **note**: receiving browser webcams requires [HTTPS/SSL](#enabling-https--ssl) to be enabled

Then navigate your browser again to `https://<JETSON-IP>:8554`, and you'll be prompted to enable access to the camera device and streaming will begin.  Until then, there will be warning messages printed in the Jetson's terminal about video capture timeouts occurring - these are normal and can be ignored, as the client providing the stream has not yet connected.

### videoSource Code

If you're using the videoSource interface in your program and want to hardcode it for receiving WebRTC (as opposed to parsing the command-line), you can create it like so:

``` python
# Python
input = jetson_utils.videoInput("webrtc://@:8554/input")

# C++
videoSource* input = videoInput::Create("webrtc://@:8554/input");
```

You can then use the videoSource interface in your main loop to capture video just like you would have before like in these [examples](aux-streaming.md#source-code).

## Full Duplex

To both send and recieve WebRTC streams simulateously, simply specify the `webrtc://` protocol for both the input and output locations:

``` bash
$ video-viewer webrtc://@:8554/input webrtc://@:8554/output  # browser->Jetson->browser loopback
$ posenet.py webrtc://@:8554/input webrtc://@:8554/output    # loopback with pose estimation
```

Then when you navigate to the page, it will both send the video from your browser's webcam and playback the results.  Subsequent examples will show how to make your own backend server applications and frontends with different web frameworks.
 
<p align="right">Next | <b><a href="webrtc-html.md">HTML / JavaScript</a></b></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
