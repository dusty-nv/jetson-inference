<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="../README.md#hello-ai-world">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Object Detection</sup></s></p>

# Collecting your own Detection Datasets

The previously used `camera-capture` tool can also label object detection datasets from live video:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-collection-detect.jpg" >

When the `Dataset Type` drop-down is in Detection mode, the tool creates datasets in [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format (which is supported during training).

> **note:** if you want to label a set of images that you already have (as opposed to capturing them from camera), try using a tool like [`CVAT`](https://github.com/openvinotoolkit/cvat) and export the dataset in Pascal VOC format.  Then create a labels.txt in the dataset with the names of each of your object classes.

## Creating the Label File

Under `jetson-inference/python/training/detection/ssd/data`, create an empty directory for storing your dataset and a text file that will define the class labels (usually called `labels.txt`).  The label file contains one class label per line, for example:

``` bash
Water
Nalgene
Coke
Diet Coke
Ginger ale
```

If you're using the container, you'll want to store your dataset in a [Mounted Directory](aux-docker.md#mounted-data-volumes) like above, so it's saved after the container shuts down.

## Launching the Tool

The `camera-capture` tool accepts the same input URI's on the command line that are found on the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page. 

Below are some example commands for launching the tool:

``` bash
$ camera-capture csi://0       # using default MIPI CSI camera
$ camera-capture /dev/video0   # using V4L2 camera /dev/video0
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

## Collecting Data

Below is the `Data Capture Control` window, after the `Dataset Type` drop-down has been set to Detection mode (do this first).

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-collection-detection-widget.jpg" >

Then, open the dataset path and class labels that you created.  The `Freeze/Edit` and `Save` buttons will then become active. 

Position the camera at the object(s) in your scene, and click the `Freeze/Edit` button (or press the spacebar).  The live camera view will then be 'frozen' and you will be able to draw bounding boxes over the objects.  You can then select the appropriate object class for each bounding box in the grid table in the control window.  When you are done labeling the image, click the depressed `Freeze/Edit` button again to save the data and unfreeze the camera view for the next image.

Other widgets in the control window include:

* `Save on Unfreeze` - automatically save the data when `Freeze/Edit` is unfreezed
* `Clear on Unfreeze` - automatically remove the previous bounding boxes on unfreeze
* `Merge Sets` - save the same data across the train, val, and test sets
* `Current Set` - select from train/val/test sets
    * for object detection, you need at least train and test sets
    * although if you check `Merge Sets`, the data will be replicated as train, val, and test
* `JPEG Quality` - control the encoding quality and disk size of the saved images

It's important that your data is collected from varying object orientations, camera viewpoints, lighting conditions, and ideally with different backgrounds to create a model that is robust to noise and changes in environment.  If you find that you're model isn't performing as well as you'd like, try adding more training data and playing around with the conditions.

## Training your Model

When you've collected a bunch of data, then you can try training a model on it using the same `train_ssd.py` script.  The training process is the same as the previous example, with the exception that the `--dataset-type=voc` and `--data=<PATH>` arguments should be set:

```bash
$ cd jetson-inference/python/training/detection/ssd
$ python3 train_ssd.py --dataset-type=voc --data=data/<YOUR-DATASET> --model-dir=models/<YOUR-MODEL>
```

> **note:** if you run out of memory or your process is "killed" during training, try [Mounting SWAP](pytorch-transfer-learning.md#mounting-swap) and [Disabling the Desktop GUI](pytorch-transfer-learning.md#disabling-the-desktop-gui). <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to save memory, you can also reduce the `--batch-size` (default 4) and `--workers` (default 2)
  
Like before, after training you'll need to convert your PyTorch model to ONNX:

```bash
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```

The converted model will then be saved under `<YOUR-MODEL>/ssd-mobilenet.onnx`, which you can then load with the `detectnet` programs like we did in the previous examples:

```bash
NET=models/<YOUR-MODEL>

detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
```

> **note:** it's important to run inference with the labels file that gets generated to your model directory, and not the one that you originally created for your dataset.  This is because a `BACKGROUND` class gets added to the class labels by `train_ssd.py` and saved to the model directory (which the trained model expects to use).

If you need to, go back and collect more training data and re-train your model again.  You can restart again and pick up where you left off using the `--resume` argument (run `python3 train_ssd.py --help` for more info).  Remember to re-export the model to ONNX after re-training.

## What's Next

This is the last step of the *Hello AI World* tutorial, which covers inferencing and transfer learning on Jetson with TensorRT and PyTorch.  

To recap, together we've covered:

* Using image recognition networks to classify images and video
* Coding your own inferencing programs in Python and C++
* Performing object detection to locate object coordinates
* Segmenting images and video with fully-convolutional networks
* Re-training models with PyTorch using transfer learning
* Collecting your own datasets and training your own models

Next we encourage you to experiment and apply what you've learned to other projects, perhaps taking advantage of Jetson's embedded form-factor - for example an autonomous robot or intelligent camera-based system.  Here are some example ideas that you could play around with:

* use GPIO to trigger external actuators or LEDs when an object is detected
* an autonomous robot that can find or follow an object
* a handheld battery-powered camera + Jetson + mini-display 
* an interactive toy or treat dispenser for your pet
* a smart doorbell camera that greets your guests

For more examples to inspire your creativity, see the **[Jetson Projects](https://developer.nvidia.com/embedded/community/jetson-projects)** page.  Good luck and have fun!

<p align="right">Back | <b><a href="pytorch-ssd.md">Re-training SSD-Mobilenet</a></p>
</b><p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
