<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-example-2.md">Back</a> | <a href="detectnet-tracking.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p>

# Using TAO Detection Models

NVIDIA's [TAO Toolkit](https://developer.nvidia.com/tao-toolkit) includes highly-accurate high-resolution object detection models, optimized/pruned and quantized for INT8 precision.  jetson-inference supports for TAO models that are based on the [DetectNet_v2](https://docs.nvidia.com/tao/tao-toolkit/text/object_detection/detectnet_v2.html) DNN architecture, including the following pre-trained models:

| Model                   | CLI argument       | Object classes       |
| ------------------------|--------------------|----------------------|
| [TAO PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet)          | `peoplenet`        | person, bag, face    |
| [TAO PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) <sup>(pruned)</sup> | `peoplenet-pruned` | person, bag, face    |
| [TAO DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet)        | `dashcamnet`       | person, car, bike, sign |
| [TAO TrafficCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/trafficcamnet)  | `trafficcamnet`    | person, car, bike, sign | 
| [TAO FaceDetect](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet)           | `facedetect`       | face                 |

Although a [section below](#importing-your-own-tao-detection-models) covers how to load your own TAO models, let's take a look at using the pre-trained models first.

### PeopleNet

[PeopleNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/peoplenet) is a high-resolution 960x544 model with up to ~90% accuracy for detecting people, bags, and faces.  It's based on DetectNet_v2 with a ResNet-34 backbone.  Launching detectnet/detectnet.py with `--model=peoplenet` will run the TAO PeopleNet model with INT8 precision on platforms that support it (FP16 otherwise).  There's also the `peoplenet-pruned` model which is faster and slightly less accurate.

<a href="https://youtu.be/rWGTxeb3Nrw" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tao-peoplenet-youtube.jpg></a>


``` bash
# Download test video
wget https://nvidia.box.com/shared/static/veuuimq6pwvd62p9fresqhrrmfqz0e2f.mp4 -O pedestrians.mp4

# C++
$ detectnet --model=peoplenet pedestrians.mp4 pedestrians_peoplenet.mp4

# Python
$ detectnet.py --model=peoplenet pedestrians.mp4 pedestrians_peoplenet.mp4
```

You can also adjust the `--confidence` and `--clustering` thresholds - these TAO models seem not introduce too many false positives with lower thresholds due to their increased accuracy.  The [Flask webapp](webrtc-flask.md) is a convenient tool for playing around with these settings interactively.

### DashCamNet

Like PeopleNet, [DashCamNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/dashcamnet) is a 960x544 detector based on DetectNet_v2 and ResNet-34.  It's intended use is for detecting people and vehicles from street-level viewpoints and first-person perspectives.  TrafficCamNet is similar, for imagery taken from a higher vantage point.

<a href="https://www.youtube.com/watch?v=tsugHIgFrwI" target="_blank"><img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tao-dashcamnet-youtube.jpg></a>

``` bash
# C++
$ detectnet --model=dashcamnet input.mp4 output.mp4

# Python
$ detectnet.py --model=dashcamnet input.mp4 output.mp4
```

> **note**: you can run this with any input/output from the [Camera Streaming and Multimedia](aux-streaming.md) page

### FaceDetect

[FaceDetect](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/facenet) is a TAO model for just detecting faces.  It was trained with up to ~85% accuracy on a dataset with more than 1.8M samples taken from a variety of camera angles.  It has a resolution of 736x416 and uses DetectNet_v2 with a ResNet-18 backbone.

<img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tao-facenet.jpg>

``` bash
# C++
$ detectnet --model=facedetect "images/humans_*.jpg" images/test/facedetect_humans_%i.jpg

# Python
$ detectnet.py --model=facedetect "images/humans_*.jpg" images/test/facedetect_humans_%i.jpg
```

### Importing Your Own TAO Detection Models

Although jetson-inference can automatically download, convert, and load the pre-trained TAO detection models above, you may wish to use a different version of those models or your own DetectNet_v2 model that you trained or fine-tuned using TAO.  To do that, copy your trained ETLT model to your Jetson, along with the appropriate version of the [`tao-converter`](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/resources/tao-converter) tool.  Then depending on your model's configuration (the details of which are typically found on the model card), you can run a script like below to generate the TensorRT engine from the ETLT:

``` bash
# model config
MODEL_DIR="peoplenet_deployable_quantized_v2.6.1"
MODEL_INPUT="$MODEL_DIR/resnet34_peoplenet_int8.etlt"
MODEL_OUTPUT="$MODEL_INPUT.engine"

INPUT_DIMS="3,544,960"
OUTPUT_LAYERS="output_bbox/BiasAdd,output_cov/Sigmoid"
MAX_BATCH_SIZE="1"

WORKSPACE="4294967296" # 4GB (default)
PRECISION="int8"       # fp32, fp16, int8
CALIBRATION="$MODEL_DIR/resnet34_peoplenet_int8.txt"

ENCRYPTION_KEY="tlt_encode"

# generate TensorRT engine
tao-converter \
	-k $ENCRYPTION_KEY \
	-d $INPUT_DIMS \
	-o $OUTPUT_LAYERS \
	-m $MAX_BATCH_SIZE \
	-w $WORKSPACE \
	-t $PRECISION \
	-c $CALIBRATION \
	-e $MODEL_OUTPUT \
	$MODEL_INPUT
```

After converting it, you can load it with detectnet/detectnet.py like so:

``` bash
$ detectnet \
	--model=$MODEL_DIR/resnet34_peoplenet_int8.etlt.engine \
	--labels=$MODEL_DIR/labels.txt \
	--input-blob=input_1 \
	--output-cvg=output_cov/Sigmoid \
	--output-bbox=output_bbox/BiasAdd \
	input.mp4 output.mp4
```

> **note**: only TAO DetectNet_v2 models are currently supported in jetson-inference, as it is setup for that network's pre/post-processing

In your own applications, you can also load them directly from [C++](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/classdetectNet.html#a9981735c38d2cb97205aa9e255ab4a0e) or [Python](https://github.com/dusty-nv/jetson-inference/blob/89a9bbe8812ec8a142910ae55e9a6c25dbdb9841/python/examples/detectnet.py#L57) by using the extended form of the detectNet API.

<p align="right">Next | <b><a href="detectnet-tracking.md">Object Tracking on Video</a></b>
<br/>
Back | <b><a href="detectnet-example-2.md">Coding Your Own Object Detection Program</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
