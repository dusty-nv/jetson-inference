<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-example-2.md">Back</a> | <a href="detectnet-tracking.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# TAO Pre-trained Detection Models

NVIDIA's [TAO Toolkit](https://developer.nvidia.com/tao-toolkit) includes highly-accurate high-resolution object detection models, optimized/pruned and quantized for INT8 precision.  jetson-inference supports for TAO models that are based on the [DetectNet_v2](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/detectnet_v2.html) DNN architecture, including the following pre-trained models that are available:

* [PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)
* [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet)
* [TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)
* [FaceDetect](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet)

Although a section below covers how to convert and load your own DetectNet_v2 model that you trained or fine-tuned with TAO, let's take a look at using the pre-trained models first.

### PeopleNet

PeopleNet is a high-resolution 960x544 model with up to ~90% accuracy for detecting people, bags, and faces.  It's based on DetectNet_v2 with a ResNet-34 backbone & feature extractor.  Running detectnet/detectnet.py with `--model=peoplenet` will download, convert, and run the TAO PeopleNet model with INT8 precision on platforms that support it (and with FP16 otherwise).  

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

There's also the `peoplenet-pruned` model ([`peoplenet_pruned_quantized_v2.3.2](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet/files?version=pruned_quantized_v2.3.2) which is faster and slightly less accurate.

<p align="right">Next | <b><a href="detectnet-tracking.md">Object Tracking</a></b>
<br/>
Back | <b><a href="detectnet-example-2.md">Coding Your Own Object Detection Program</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
