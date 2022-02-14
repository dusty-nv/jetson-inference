<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-camera-2.md">Back</a> | <a href="depthnet.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Pose Estimation</sup></s></p>

# Pose Estimation with PoseNet
Pose estimation consists of locating various body parts (aka keypoints) that form a skeletal topology (aka links). Pose estimation has a variety of applications including gestures, AR/VR, HMI (human/machine interface), and posture/gait correction. [Pre-trained models](#pre-trained-pose-estimation-models) are provided for human body and hand pose estimation that are capable of detecting multiple people per frame.  

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-0.jpg">

The [`poseNet`](../c/poseNet.h) object accepts an image as input, and outputs a list of object poses.  Each object pose contains a list of detected keypoints, along with their locations and links between keypoints.  You can query these to find particular features.  [`poseNet`](../c/poseNet.h) can be used from [Python](https://rawgit.com/dusty-nv/jetson-inference/dev/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h).

As examples of using the `poseNet` class, we provide sample programs for C++ and Python:

- [`posenet.cpp`](../examples/posenet/posenet.cpp) (C++) 
- [`posenet.py`](../python/examples/posenet.py) (Python) 

These samples are able to detect the poses of multiple humans in images, videos, and camera feeds.  For more info about the various types of input/output streams supported, see the [Camera Streaming and Multimedia](aux-streaming.md) page.

## Pose Estimation on Images

First, let's try running the `posenet` sample on some examples images.  In addition to the input/output paths, there are some additional command-line options that are optional:

- optional `--network` flag which changes the [pose model](#pre-trained-pose-estimation-models) being used (the default is `resnet18-body`).
- optional `--overlay` flag which can be comma-separated combinations of `box`, `links`, `keypoints`, and `none`
	- The default is `--overlay=links,keypoints` which displays circles over they keypoints and lines over the links
- optional `--keypoint-scale` value which controls the radius of the keypoint circles in the overlay (the default is `0.0052`)
- optional `--link-scale` value which controls the line width of the link lines in the overlay (the default is `0.0013`)
- optional `--threshold` value which sets the minimum threshold for detection (the default is `0.15`).  

If you're using the [Docker container](aux-docker.md), it's recommended to save the output images to the `images/test` mounted directory.  These images will then be easily viewable from your host device under `jetson-inference/data/images/test` (for more info, see [Mounted Data Volumes](aux-docker.md#mounted-data-volumes)). 

Here are some examples of human pose estimation using the default Pose-ResNet18-Body model:

``` bash
# C++
$ ./posenet "images/humans_*.jpg" images/test/pose_humans_%i.jpg

# Python
$ ./posenet.py "images/humans_*.jpg" images/test/pose_humans_%i.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-1.jpg">

> **note**:  the first time you run each model, TensorRT will take a few minutes to optimize the network. <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;this optimized network file is then cached to disk, so future runs using the model will load faster.

There are also test images of people under `"images/peds_*.jpg"` that you can try as well.

## Pose Estimation from Video 

To run pose estimation on a live camera stream or video, pass in a device or file path from the [Camera Streaming and Multimedia](aux-streaming.md) page.

``` bash
# C++
$ ./posenet /dev/video0     # csi://0 if using MIPI CSI camera

# Python
$ ./posenet.py /dev/video0  # csi://0 if using MIPI CSI camera
```

<a href="https://www.youtube.com/watch?v=hwFtWYR986Q" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-video-body.jpg width="750"></a>

``` bash
# C++
$ ./posenet --network=resnet18-hand /dev/video0

# Python
$ ./posenet.py --network=resnet18-hand /dev/video0
```

<a href="https://www.youtube.com/watch?v=6NL_IE44vRE" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/posenet-video-hands.jpg width="750"></a>

## Pre-trained Pose Estimation Models

Below are the pre-trained pose estimation networks available for [download](building-repo-2.md#downloading-models), and the associated `--network` argument to `posenet` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Keypoints |
| ------------------------|--------------------|--------------------|-----------|
| Pose-ResNet18-Body      | `resnet18-body`    | `RESNET18_BODY`    | 18        |
| Pose-ResNet18-Hand      | `resnet18-hand`    | `RESNET18_HAND`    | 21        |
| Pose-DenseNet121-Body   | `densenet121-body` | `DENSENET121_BODY` | 18        |

> **note**:  to download additional networks, run the [Model Downloader](building-repo-2.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, Pose-ResNet18-Body is used if the optional `--network` flag isn't specified.


## Working with Object Poses

If you want to access the pose keypoint locations, the `poseNet.Process()` function returns a list of `poseNet.ObjectPose` structures.  Each object pose represents one object (i.e. one person) and contains a list of detected keypoints and links - see the [Python](https://rawgit.com/dusty-nv/jetson-inference/python/docs/html/python/jetson.inference.html#poseNet) and [C++](../c/poseNet.h) docs for more info.  

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
Back | <b><a href="segnet-camera-2.md">Running the Live Camera Segmentation Demo</a></p>
</b><p align="center"><sup>© 2016-2021 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
