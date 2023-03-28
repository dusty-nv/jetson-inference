<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-tao.md">Back</a> | <a href="segnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Object Tracking on Video

Although with the accuracy of modern detection DNNs you can essentially do "tracking by detection", some degree of temporal filtering can be beneficial to smooth over blips in the detections and temporary occlusions in the video.  jetson-inference includes basic (but fast) multi-object tracking using frame-to-frame IOU (intersection-over-union) bounding box comparisons from [`High-Speed Tracking-by-Detection Without Using Image Information`](http://elvera.nue.tu-berlin.de/typo3/files/1517Bochinski2017.pdf) (DeepStream has more comprehensive tracking implementations available [from here](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)).

<a href="https://www.youtube.com/watch?v=L8vwuXKQrow" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tracking-pedestrians-youtube.jpg></a>

To enable tracking with detectnet/detectnet.py, run it with the `--tracking` flag.  

``` bash
# Download test video
wget https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4 -O pedestrians.mp4

# C++
$ detectnet --model=peoplenet --tracking pedestrians.mp4 pedestrians_tracking.mp4

# Python
$ detectnet.py --model=peoplenet --tracking pedestrians.mp4 pedestrians_tracking.mp4
```

### Tracking Parameters 

There are other tracker settings you can change with the following command-line options:

```
objectTracker arguments:
  --tracking               flag to enable default tracker (IOU)
  --tracker-min-frames=N   the number of re-identified frames for a track to be considered valid (default: 3)
  --tracker-drop-frames=N  number of consecutive lost frames before a track is dropped (default: 15)
  --tracker-overlap=N      how much IOU overlap is required for a bounding box to be matched (default: 0.5)
```

These settings can be also made via Python with the [`detectNet.SetTrackerParams()`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet) function or in C++ by the [`objectTracker`](../c/tracking/objectTrackerIOU.h) API:

#### Python
``` Python
from jetson_utils import detectNet

net = detectNet()

net.SetTrackingEnabled(True)
net.SetTrackingParams(minFrames=3, dropFrames=15, overlapThreshold=0.5)
```

#### C++
``` cpp
#include <jetson-inference/detectNet.h>
#include <jetson-inference/objectTrackerIOU.h>

detectNet* net = detectNet::Create();

net->SetTracker(objectTrackerIOU::Create(3, 15, 0.5f));
``` 

To play around with these settings interactively, you can use the [Flask webapp](webrtc-flask.md) from your browser.

### Tracking State

When tracking is enabled, the [`detectNet.Detection`](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/structdetectNet_1_1Detection.html) results that are returned from Detect() will have additional variables activated that get updated by the tracker and describe the tracking state of each object - such as TrackID, TrackStatus, TrackFrames, and TrackLost:

``` cpp
struct Detection
{
	// Detection Info
	uint32_t ClassID;	/**< Class index of the detected object. */
	float Confidence;	/**< Confidence value of the detected object. */

	// Tracking Info
	int TrackID;		/**< Unique tracking ID (or -1 if untracked) */
	int TrackStatus;	/**< -1 for dropped, 0 for initializing, 1 for active/valid */ 
	int TrackFrames;	/**< The number of frames the object has been re-identified for */
	int TrackLost;   	/**< The number of consecutive frames tracking has been lost for */
	
	// Bounding Box Coordinates
	float Left;		/**< Left bounding box coordinate (in pixels) */
	float Right;		/**< Right bounding box coordinate (in pixels) */
	float Top;		/**< Top bounding box cooridnate (in pixels) */
	float Bottom;		/**< Bottom bounding box coordinate (in pixels) */
};
```

These are accessible from both [C++](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/structdetectNet_1_1Detection.html) and [Python](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/python/jetson.inference.html#detectNet).  For example, from Python:

``` python
detections = net.Detect(img)

for detection in detections:
    if detection.TrackStatus >= 0:  # actively tracking
        print(f"object {detection.TrackID} at ({detection.Left}, {detection.Top}) has been tracked for {detection.TrackFrames} frames")
    else:  # if tracking was lost, this object will be dropped the next frame
        print(f"object {detection.TrackID} has lost tracking")   
```

If the track was lost (`TrackStatus=-1`), that object will be dropped and no longer included in the detections array on subsequent frames.

<p align="right">Next | <b><a href="segnet-console-2.md">Semantic Segmentation</a></b>
<br/>
Back | <b><a href="detectnet-tao.md">Using TAO Detection Models</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
