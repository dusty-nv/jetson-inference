<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="pytorch-ssd.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Classification</sup></s></p>

# Collecting your own Classification Datasets

In order to collect your own datasets for training customized models to classify objects or scenes of your choosing, we've created an easy-to-use tool called `camera-capture` for capturing and labeling images on your Jetson from live video:

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-collection.jpg" >

The tool will create datasets with the following directory structure on disk:

```
‣ train/
	• class-A/
	• class-B/
	• ...
‣ val/
	• class-A/
	• class-B/
	• ...
‣ test/
	• class-A/
	• class-B/
	• ...
```

where `class-A`, `class-B`, ect. will be subdirectories containing the data for each object class that you've defined in a class label file.  The names of these class subdirectories will match the class label names that we'll create below.  These subdirectories will automatically be populated by the tool for the `train`, `val`, and `test` sets from the classes listed in the label file, and a sequence of JPEG images will be saved under each.

Note that above is the organization structure expected by the PyTorch training script that we've been using.  If you inspect the Cat/Dog and PlantCLEF datasets, they're also organized in the same way.

## Creating the Label File

Under `jetson-inference/python/training/classification/data`, create an empty directory for storing your dataset and a text file that will define the class labels (usually called `labels.txt`).  The label file contains one class label per line, and is alphabetized (this is important so the ordering of the classes in the label file matches the ordering of the corresponding subdirectories on disk).  As mentioned above, the `camera-capture` tool will automatically populate the necessary subdirectories for each class from this label file.

Here's an example `labels.txt` file with 5 classes:

``` bash
background
brontosaurus
tree
triceratops
velociraptor
```

And here's the corresponding directory structure that the tool will create:

``` bash
‣ train/
	• background/
	• brontosaurus/
	• tree/
	• triceratops/
	• velociraptor/
‣ val/
	• background/
	• brontosaurus/
	• tree/
	• triceratops/
	• velociraptor/
‣ test/
	• background/
	• brontosaurus/
	• tree/
	• triceratops/
	• velociraptor/
```

If you're using the container, you'll want to store your dataset in a [Mounted Directory](aux-docker.md#mounted-data-volumes) like above, so it's saved after the container shuts down.

## Launching the Tool

The source for the `camera-capture` tool can be found under [`jetson-inference/tools/camera-capture/`](https://github.com/dusty-nv/camera-capture), and like the other programs from the repo it gets built to the `aarch64/bin` directory and installed under `/usr/local/bin/`  

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

Below is the `Data Capture Control` window, which allows you to pick the desired path to the dataset and load the class label file that you created above, and then presents options for selecting the current object class and train/val/test set that you are currently collecting data for:

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-collection-widget.jpg" >

First, open the dataset path and class labels.  The tool will then create the dataset structure discussed above (unless these subdirectories already exist), and you will see your object labels populated inside the `Current Class` drop-down.  Leave the `Dataset Type` as Classification.

Then position the camera at the object or scene you have currently selected in the drop-down, and click the `Capture` button (or press the spacebar) when you're ready to take an image.  The images will be saved under that class subdirectory in the train, val, or test set.  The status bar displays how many images have been saved under that category.

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-capture-brontosaurus.gif" >

It's recommended to collect at least 100 training images per class before attempting training.  A rule of thumb for the validation set is that it should be roughly 10-20% the size of the training set, and the size of the test set is simply dictated by how many static images you want to test on.  You can also just run the camera to test your model if you'd like.

It's important that your data is collected from varying object orientations, camera viewpoints, lighting conditions, and ideally with different backgrounds to create a model that is robust to noise and changes in environment.  If you find that you're model isn't performing as well as you'd like, try adding more training data and playing around with the conditions.


## Training your Model

When you've collected a bunch of data, then you can try training a model on it, just like we've done before.  The training process is the same as the previous examples, and the same PyTorch scripts are used:

```bash
$ cd jetson-inference/python/training/classification
$ python3 train.py --model-dir=models/<YOUR-MODEL> data/<YOUR-DATASET>
```

> **note:** if you run out of memory or your process is "killed" during training, try [Mounting SWAP](pytorch-transfer-learning.md#mounting-swap) and [Disabling the Desktop GUI](pytorch-transfer-learning.md#disabling-the-desktop-gui). <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to save memory, you can also reduce the `--batch-size` (default 8) and `--workers` (default 2)
  
Like before, after training you'll need to convert your PyTorch model to ONNX:

```bash
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```

The converted model will be saved under `models/<YOUR-MODEL>/resnet18.onnx`, which you can then load with the `imagenet` programs like we did in the previous examples:

```bash
NET=models/<YOUR-MODEL>
DATASET=data/<YOUR-DATASET>

# C++ (MIPI CSI)
imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0

# Python (MIPI CSI)
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0
```

If you need to, go back and collect more data and re-train your model again.  You can restart the training from where you left off using the `--resume` and `--epoch-start` flags (run `python3 train.py --help` for more info).  Then remember to re-export the model.

Next, we're going to train our own object detection models with PyTorch.

<p align="right">Next | <b><a href="pytorch-ssd.md">Re-training SSD-Mobilenet</a></b>
<br/>
Back | <b><a href="pytorch-plants.md">Re-training on the PlantCLEF Dataset</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
