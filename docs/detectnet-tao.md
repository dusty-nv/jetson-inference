<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-example-2.md">Back</a> | <a href="detectnet-tracking.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# TAO Detection Models

NVIDIA's [TAO Toolkit](https://developer.nvidia.com/tao-toolkit) includes highly-accurate high-resolution object detection models, optimized/pruned and quantized for INT8 precision.  jetson-inference supports for TAO models that are based on the [DetectNet_v2](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/detectnet_v2.html) DNN architecture, including the following pre-trained models:

| Model                   | CLI argument       | Object classes       |
| ------------------------|--------------------|----------------------|
| TAO [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)          | `peoplenet`        | person, bag, face    |
| TAO [PeopleNet (pruned)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) | `peoplenet-pruned` | person, bag, face    |
| TAO [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet)        | `dashcamnet`       | person, car, bike, sign |
| TAO [TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)  | `trafficcamnet`    | person, car, bike, sign | 
| TAO [FaceDetect](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet)           | `facedetect`       | face                 |

Although a section below covers how to load your own TAO models, let's take a look at using the pre-trained models first.

### PeopleNet

[PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) is a high-resolution 960x544 model with up to ~90% accuracy for detecting people, bags, and faces.  It's based on DetectNet_v2 with a ResNet-34 backbone & feature extractor.  Launching detectnet/detectnet.py with `--model=peoplenet` will run the TAO PeopleNet model with INT8 precision on platforms that support it (FP16 otherwise).  There's also the `peoplenet-pruned` model ([`peoplenet_pruned_quantized_v2.3.2`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet/files?version=pruned_quantized_v2.3.2) which is faster and slightly less accurate.

<a href="https://youtu.be/rWGTxeb3Nrw" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tao-peoplenet-youtube.jpg></a>


``` bash
# Download test video
wget https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4 -O pedestrians.mp4

# C++
$ ./detectnet --model=peoplenet pedestrians.mp4 pedestrians_peoplenet.mp4

# Python
$ ./detectnet.py --model=peoplenet pedestrians.mp4 pedestrians_peoplenet.mp4
```

> **note**: you can run this with any input/output from the [Camera Streaming and Multimedia](aux-streaming.md) page

You can also adjust the `--confidence` and `--clustering` thresholds - due to their increased accuracy, these TAO models seem to behave well with lowered thresholds without introducing too many false positives.  The [Flask webapp](#webrtc-flask.md) is a convenient tool for playing around with these settings interactively.

### DashCamNet

Like PeopleNet, [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet) is a 960x544 detector based on DetectNet_v2 and ResNet-34.  It's intended use is for detecting people and vehicles from street-level viewpoints and first-person perspectives.

<a href="https://www.youtube.com/watch?v=tsugHIgFrwI" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tao-dashcamnet-youtube.jpg></a>

``` bash
# C++
$ ./detectnet --model=dashcamnet input.mp4 output.mp4

# Python
$ ./detectnet.py --model=dashcamnet input.mp4 output.mp4
```

TrafficCamNet is similar, for imagery taken from a higher vantage point.


<p align="right">Next | <b><a href="detectnet-tracking.md">Object Tracking</a></b>
<br/>
Back | <b><a href="detectnet-example-2.md">Coding Your Own Object Detection Program</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
