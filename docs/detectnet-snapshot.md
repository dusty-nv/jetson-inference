<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-training.md">Back</a> | <a href="detectnet-console.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p> 

# Downloading the Detection Model to Jetson

Next, download and extract the trained model snapshot to Jetson.  From the browser on your Jetson TX1/TX2, navigate to your DIGITS server and the `DetectNet-COCO-Dog` model.  Under the `Trained Models` section, select the desired snapshot from the drop-down (usually the one with the highest epoch) and click the `Download Model` button.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-model-download-dog.png" width="650">

Alternatively, if your Jetson and DIGITS server aren't accessible from the same network, you can use the step above to download the snapshot to an intermediary machine and then use SCP or USB stick to copy it to Jetson.  

Then extract the archive with a command similar to:

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170504-190602-879f_epoch_100.0.tar.gz
```

### DetectNet Patches for TensorRT

In the original DetectNet prototxt exists a Python clustering layer which isn't available in TensorRT and should be deleted from the `deploy.prototxt` included in the snapshot.  In this repo the [`detectNet`](detectNet.h) class handles the clustering as opposed to Python.

At the end of `deploy.prototxt`, delete the layer named `cluster`:

```
layer {
  name: "cluster"
  type: "Python"
  bottom: "coverage"
  bottom: "bboxes"
  top: "bbox-list"
  python_param {
    module: "caffe.layers.detectnet.clustering"
    layer: "ClusterDetections"
    param_str: "640, 640, 16, 0.6, 2, 0.02, 22, 1"
  }
}
```

Without this Python layer, the snapshot can now be imported into TensorRT onboard the Jetson.

##
<p align="right">Next | <b><a href="detectnet-console.md">Detecting Objects from the Command Line</a></b>
<br/>
Back | <b><a href="detectnet-training.md">Locating Object Coordinates using DetectNet</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
