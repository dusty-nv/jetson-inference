<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-camera-2.md">Back</a> | <a href="detectnet-tao.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Coding Your Own Object Detection Program

In this step of the tutorial, we'll walk through the creation of the previous example for realtime object detection on a live camera feed in only 10 lines of Python code.  The program will load the detection network with the [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet) object, capture video frames and process them, and then render the detected objects to the display.

For your convenience and reference, the completed source is available in the [`python/examples/my-detection.py`](../python/examples/my-detection.py) file of the repo, but the guide below will act like they reside in the user's home directory or in an arbitrary directory of your choosing.  

Here's a quick preview of the Python code we'll be walking through:

``` python
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

There's also a video screencast of this coding tutorial available on YouTube:

<a href="https://www.youtube.com/watch?v=obt60r8ZeB0&list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8&index=12" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/thumbnail_detectnet.jpg width="750"></a>
## Source Code

First, open up your text editor of choice and create a new file.  Below we'll assume that you'll save it on your host device under your user's home directory as `~/my-detection.py`, but you can name and store it where you wish.  If you're using the Docker container, you'll want to store your code in a [Mounted Directory](aux-docker.md#mounted-data-volumes), similar to what we did in the [Image Recognition Python Example](imagenet-example-python-2.md#setting-up-the-project).

#### Importing Modules

At the top of the source file, we'll import the Python modules that we're going to use in the script.  Add `import` statements to load the [`jetson.inference`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html) and [`jetson.utils`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html) modules used for object detection and camera capture.

``` python
import jetson.inference
import jetson.utils
```

> **note**:  these Jetson modules are installed during the `sudo make install` step of [building the repo](building-repo-2.md#compiling-the-project).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if you did not run `sudo make install`, then these packages won't be found when the example is run.  

#### Loading the Detection Model

Next use the following line to create a [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet) object instance that loads the [91-class](../data/networks/ssd_coco_labels.txt) SSD-Mobilenet-v2 model:

``` python
# load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
```

Note that you can change the model string to one of the values from [this table](detectnet-console-2.md#pre-trained-detection-models-available) to load a different detection model.  We also set the detection threshold here to the default of `0.5` for illustrative purposes - you can tweak it later if needed.

#### Opening the Camera Stream

To connect to the camera device for streaming, we'll create an instance of the [`videoSource`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html#videoSource) object:

``` python
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
```

The string passed to `videoSource()` can actually be any valid resource URI, whether it be a camera, video file, or network stream.  For more information about video streams and protocols, please see the [Camera Streaming and Multimedia](aux-streaming.md) page.

> **note**:  for compatible cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Display Loop

Next, we'll create a video output interface with the [`videoOutput`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.utils.html#videoOutput) object and create a main loop that will run until the user exits:

``` python
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	# main loop will go here
```

Note that the remainder of the code below should be indented underneath this `while` loop.  Similar to above, you can substitute the URI string for other types of outputs found on [this page](aux-streaming.md) (like video files, ect).

#### Camera Capture

The first thing that happens in the main loop is to capture the next video frame from the camera.  `camera.Capture()` will wait until the next frame has been sent from the camera and loaded into GPU memory.

``` python
	img = camera.Capture()
```

The returned image will be a [`jetson.utils.cudaImage`](aux-image.md#image-capsules-in-python) object that contains attributes like width, height, and pixel format:

```python
<jetson.utils.cudaImage>
  .ptr      # memory address (not typically used)
  .size     # size in bytes
  .shape    # (height,width,channels) tuple
  .width    # width in pixels
  .height   # height in pixels
  .channels # number of color channels
  .format   # format string
  .mapped   # true if ZeroCopy
```

For more information about accessing images from Python, see the [Image Manipulation with CUDA](aux-image.md) page.  

#### Detecting Objects

Next the detection network processes the image with the `net.Detect()` function.  It takes in the image from `camera.Capture()` and returns a list of detections:

``` python
	detections = net.Detect(img)
```

This function will also automatically overlay the detection results on top of the input image.

If you want, you can add a `print(detections)` statement here, and the coordinates, confidence, and class info will be printed out to the terminal for each detection result.  Also see the [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet) documentation for info about the different members of the `Detection` structures that are returned for accessing them directly in a custom application.

#### Rendering

Finally we'll visualize the results with OpenGL and update the title of the window to display the current peformance:

``` python
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

The `Render()` function will automatically flip the backbuffer and present the image on-screen.

#### Source Listing

That's it!  For completness, here's the full source of the Python script that we just created:

``` python
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("csi://0")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
	img = camera.Capture()
	detections = net.Detect(img)
	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

Note that this version assumes you are using a MIPI CSI camera.  See the [`Opening the Camera Stream`](#opening-the-camera-stream) section above for info about changing it to use a different kind of input.

## Running the Program

To run the application we just coded, simply launch it from a terminal with the Python interpreter:

``` bash
$ python3 my-detection.py
```

To tweak the results, you can try changing the model that's loaded along with the detection threshold.  Have fun!

<p align="right">Next | <b><a href="detectnet-tao.md">Using TAO Detection Models</a></b>
<br/>
Back | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
