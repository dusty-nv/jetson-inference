<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="detectnet-camera-2.md">Back</a> | <a href="segnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Coding Your Own Object Detection Program

In this step of the tutorial, we'll walk through creating an application in only 10 lines of Python code for realtime object detection on a live camera feed called [`my-detection.py`](../python/examples/my-detection.py).  The program will load the detection network with the [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) object, capture video frames and process them, and then render the detected objects to the display.

For your convenience and reference, the completed source is available in the [`python/examples/my-detection.py`](../python/examples/my-detection.py) file of the repo, but the guide below will act like they reside in the user's home directory or in an arbitrary directory of your choosing.

## Source Code

First, open up your text editor of choice and create a new file.  Below we'll assume that you'll save it to your user's home directory as `~/my-detection.py`, but you can name and store it where you wish.

First, we'll import the Python modules that we're going to use in the script.

#### Importing Modules

Add `import` statements to load the [`jetson.inference`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html) and [`jetson.utils`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.utils.html) modules used for object detection and camera capture.

``` python
import jetson.inference
import jetson.utils
```

> **note**:  these Jetson modules are installed during the `sudo make install` step of [building the repo](building-repo-2.md#compiling-the-project).  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if you did not run `sudo make install`, then these packages won't be found when we go to run the example.  

#### Loading the Detection Model

Next use the following line to create a [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) instance that loads the SSD-Mobilenet-v2 model:

``` python
# load the object detection model
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
```

Note that you can change the model string to one of the values from [this table](detectnet-console-2.md#pre-trained-detection-models-available) to load a different model.  We also set the detection threshold here to the default of `0.5` for illustrative purposes.  You can tweak it later as need be.

#### Opening the Camera Stream

To connect to the camera device for streaming, next we'll create an instance of the [`gstCamera`](https://rawgit.com/dusty-nv/jetson-inference/pytorch/docs/html/python/jetson.utils.html#gstCamera) object.  It's constructor accepts 3 parameters - the desired width, height, and video device to use.  Use the following snippet depending on if you are using a MIPI CSI camera or a V4L2 USB camera:

- MIPI CSI cameras are used by specifying the sensor index (`"0"` or `"1"`, ect.)
	- `camera = jetson.utils.gstCamera(1280, 1024, "0")`
- V4L2 USB cameras are used by specifying their `/dev/video` node (`"/dev/video0"`, `"/dev/video1"`, ect.)
	- `camera = jetson.utils.gstCamera(1280, 1024, "/dev/video0")`
- The width and height should be a resolution that the camera supports.
     - Query the available resolutions with the following commands:  
          ``` bash
          $ sudo apt-get install v4l-utils
          $ v4l2-ctl --list-formats-ext
          ```
	- If needed, change `1280` and `1024` above to the desired width/height

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

#### Display Loop

Next, we'll create an OpenGL display with the ([`glDisplay`](https://rawgit.com/dusty-nv/jetson-inference/pytorch/docs/html/python/jetson.utils.html#gstCamera) object and create a main loop that will run until the user exits the window:

``` python
display = jetson.utils.glDisplay()

while display.IsOpen():
	# main loop will go here
```

The remainder of the code below should be indented underneath this `while` loop.

#### Camera Capture

The first thing that happens in the main loop is to capture the next video frame from the camera.  `camera.CaptureRGBA()` will wait until the next frame has been sent from the camera, and after it's been acquired convert it into RGBA floating-point format on the GPU.

``` python
	img, width, height = camera.CaptureRGBA()
```

Returned are a tuple containing a reference to the image data on the GPU, along with it's dimensions.

#### Detecting Objects

Next the detection network processes the image with the `net.Detect()` function.  It takes in the image, width, and height from `camera.CaptureRGBA()` and returns a list of detections:

``` python
	detections = net.Detect(img, width, height)
```

This function will also automatically overlay the detection results on top of the input image.

If you want, you can add a `print(detections)` statement here, and the coordinates, confidence, and class info will be printed out to the terminal for each detection.  Also see the [`detectNet`](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#detectNet) documentation for info about the different members of the `Detection` structures that are returned for accessing them directly in a custom application.

#### Rendering

Finally we'll visualize the results with OpenGL and update the title of the window to display the current peformance:

``` python
	display.RenderOnce(img, width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

The `RenderOnce()` function will automatically flip the backbuffer and is used when we only have one image to render.

#### Source Listing

That's it!  For completness, here is the full source of the Python script that we just created.  You can also find it in the repo at [`python/examples/my-detection.py`](../python/examples/my-detection.py)

``` python
import jetson.inference
import jetson.utils

net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.gstCamera(1280, 1024, "/dev/video0")  # using V4L2
display = jetson.utils.glDisplay()

while display.IsOpen():
	img, width, height = camera.CaptureRGBA()
	detections = net.Detect(img, width, height)
	display.RenderOnce(img, width, height)
	display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
```

Note that this version aboves assumes you are using a V4L2 USB camera.  See the [`Opening the Camera Stream`](#opening-the-camera-stream) section above for info about changing it to use a MIPI CSI camera or supporting different resolutions.

## Running the Program

To run the application we just coded, simply launch it from a terminal with the Python interpreter:

``` bash
$ python my-detection.py
```

To tweak the results, you can try changing the model that's loaded along with the detection threshold.  Have fun!

<p align="right">Next | <b><a href="segnet-console-2.md">Semantic Segmentation with SegNet</a></b>
<br/>
Back | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
