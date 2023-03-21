<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="posenet.md">Back</a> | <a href="backgroundnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Action Recognition</sup></s></p>

# Action Recognition
Action recognition classifies the activity, behavior, or gesture occuring over a sequence of video frames.  The DNNs typically use image classification backbones with an added temporal dimension.  For example, the ResNet18-based pre-trained models use a window of 16 frames.  You can also skip frames to lengthen the window of time over which the model classifies actions.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/actionnet-windsurfing.gif">

The [`actionNet`](../c/actionNet.h) object takes in one video frame at a time, buffers them as input to the model, and outputs the class with the highest confidence.  [`actionNet`](../c/actionNet.h) can be used from [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#actionNet) and [C++](../c/actionNet.h).

As examples of using the `actionNet` class, there are sample programs for C++ and Python:

- [`actionnet.cpp`](../examples/actionnet/actionnet.cpp) (C++) 
- [`actionnet.py`](../python/examples/actionnet.py) (Python) 

## Running the Example

To run action recognition on a live camera stream or video, pass in a device or file path from the [Camera Streaming and Multimedia](aux-streaming.md) page.

``` bash
# C++
$ ./actionnet /dev/video0           # V4L2 camera input, display output (default) 
$ ./actionnet input.mp4 output.mp4  # video file input/output (mp4, mkv, avi, flv)

# Python
$ ./actionnet.py /dev/video0           # V4L2 camera input, display output (default) 
$ ./actionnet.py input.mp4 output.mp4  # video file input/output (mp4, mkv, avi, flv)
```

### Command-Line Arguments

These optional command-line arguments can be used with actionnet/actionnet.py:

```
  --network=NETWORK    pre-trained model to load, one of the following:
                           * resnet-18 (default)
                           * resnet-34
  --model=MODEL        path to custom model to load (.onnx)
  --labels=LABELS      path to text file containing the labels for each class
  --input-blob=INPUT   name of the input layer (default is 'input')
  --output-blob=OUTPUT name of the output layer (default is 'output')
  --threshold=CONF     minimum confidence threshold for classification (default is 0.01)
  --skip-frames=SKIP   how many frames to skip between classifications (default is 1)
```

By default, the model will process every-other frame to lengthen the window of time for classifying actions over.  You can change this with the `--skip-frames` parameter (using `--skip-frames=0` will process every frame).

### Pre-trained Action Recognition Models

Below are the pre-trained action recognition model available, and the associated `--network` argument to `actionnet` used for loading them:

| Model                    | CLI argument | Classes |
| -------------------------|--------------|---------|
| Action-ResNet18-Kinetics | `resnet18`   |  1040   |
| Action-ResNet34-Kinetics | `resnet34`   |  1040   |

The default is `resnet18`.  These models were trained on the [Kinetics 700](https://www.deepmind.com/open-source/kinetics) and [Moments in Time](http://moments.csail.mit.edu/) datasets (see [here](https://gist.github.com/dusty-nv/3aaa2494f7be212391cca1927ef7c74e) for the list of class labels).

##
<p align="right">Next | <b><a href="backgroundnet.md">Background Removal</a></b>
<br/>
Back | <b><a href="posenet.md">Pose Estimation with PoseNet</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
