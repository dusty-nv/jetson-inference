<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="posenet.md">Back</a> | <a href="depthnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Action Recognition</sup></s></p>

# Action Recognition
Action recognition classifies the activity or behavior occuring over a sequence of video frames.  The DNNs typically use image classification backbones with an added temporal dimension.  For example, the ResNet18-based pre-trained models use a window of 16 frames.  You can also skip frames to lengthen the window of time over which the model classifies actions.

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

By default, the model will process every-other frame.  You can lengthen or shorten this with the `--skip-frames` parameter (`--skip-frames=0` will process every frame).

### Pre-trained Action Recognition Models

Below are the pre-trained action recognition model available, and the associated `--network` argument to `actionnet` used for loading them:

| Model                   | CLI argument       | NetworkType enum   | Keypoints |
| ------------------------|--------------------|--------------------|-----------|
| Pose-ResNet18-Body      | `resnet18-body`    | `RESNET18_BODY`    | 18        |
| Pose-ResNet18-Hand      | `resnet18-hand`    | `RESNET18_HAND`    | 21        |
| Pose-DenseNet121-Body   | `densenet121-body` | `DENSENET121_BODY` | 18        |

These were trained on the Kinetics dataset 

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, Pose-ResNet18-Body is used if the optional `--network` flag isn't specified.


## Working with Object Poses

If you want to access the pose keypoint locations, the `poseNet.Process()` function returns a list of `poseNet.ObjectPose` structures.  Each object pose represents one object (i.e. one person) and contains a list of detected keypoints and links - see the [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h) docs for more info.  

Below is Python pseudocode for finding the 2D direction (in image space) that a person is pointing, by forming a vector between the `left_shoulder` and `left_wrist` keypoints:

``` python
poses = net.Process(img)

for pose in poses:
    # find the keypoint index from the list of detected keypoints
    # you can find these keypoint names in the model's JSON file, 
    # or with net.GetKeypointName() / net.GetNumKeypoints()
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    left_shoulder_idx = pose.FindKeypoint('left_shoulder')

    # if the keypoint index is < 0, it means it wasn't found in the image
    if left_wrist_idx < 0 or left_shoulder_idx < 0:
        continue
	
    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

    point_x = left_shoulder.x - left_wrist.x
    point_y = left_shoulder.y - left_wrist.y

    print(f"person {pose.ID} is pointing towards ({point_x}, {point_y})")
```
	
This was a simple example, but you can make it more useful with further manipulation of the vectors and by looking up more keypoints.  There are also more advanced techniques that use machine learning on the pose results for gesture classification, like in the [`trt_hand_pose`](https://github.com/NVIDIA-AI-IOT/trt_pose_hand) project.
	
	
##
<p align="right">Next | <b><a href="depthnet.md">Monocular Depth Estimation</a></b>
<br/>
Back | <b><a href="posenet.md">Pose Estimation with PoseNet</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
