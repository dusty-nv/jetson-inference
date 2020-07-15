<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="pytorch-ssd.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Classification</sup></s></p>

# Collecting your own Datasets

In order to collect your own datasets for training customized models to classify objects or scenes of your choosing, we've created an easy-to-use tool called `camera-capture` for capturing and labelling images on your Jetson from live video:

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

First, create an empty directory for storing your dataset and a text file that will define the class labels (usually called `labels.txt`).  The label file contains one class label per line, and is alphabetized (this is important so the ordering of the classes in the label file matches the ordering of the corresponding subdirectories on disk).  As mentioned above, the `camera-capture` tool will automatically populate the necessary subdirectories for each class from this label file.

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

Next, we'll cover the command-line options for starting the tool.

## Launching the Tool

The source for the `camera-capture` tool can be found under [`jetson-inference/tools/camera-capture/`](../tools/camera-capture), and like the other programs from the repo it gets built to the `aarch64/bin` directory and installed under `/usr/local/bin/`  

The `camera-capture` tool accepts the same input URI's on the command line that are found on the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page. 

Below are some example commands for launching the tool:

``` bash
$ camera-capture csi://0      # using default MIPI CSI camera
$ camera-capture /dev/video0  # using V4L2 camera /dev/video0
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

## Collecting Data

Below is the `Data Capture Control` window, which allows you to pick the desired path to the dataset and load the class label file that you created above, and then presents options for selecting the current object class and train/val/test set that you are currently collecting data for:

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-collection-widget.jpg" >

First, open the dataset path and class labels.  The tool will then create the dataset structure discussed above (unless these subdirectories already exist), and you will see your object labels populated inside the `Current Class` drop-down.  

Then position the camera at the object or scene you have currently selected in the drop-down, and click the `Capture` button (or press the spacebar) when you're ready to take an image.  The images will be saved under that class subdirectory in the train, val, or test set.  The status bar displays how many images have been saved under that category.

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-capture-brontosaurus.gif" >

It's recommended to collect at least 100 training images per class before attempting training.  A rule of thumb for the validation set is that it should be roughly 10-20% the size of the training set, and the size of the test set is simply dictated by how many static images you want to test on.  You can also just run the camera to test your model if you'd like.

It's important that your data is collected from varying object orientations, camera viewpoints, lighting conditions, and ideally with different backgrounds to create a model that is robust to noise and changes in environment.  If you find that you're model isn't performing as well as you'd like, try adding more training data and playing around with the conditions.


## Training your Model

When you've collected a bunch of data, then you can try training a model on it, just like we've done before.  The training process is the same as the previous examples, and the same PyTorch scripts are used:

```bash
$ cd jetson-inference/python/training/classification
$ python train.py --model-dir=<YOUR-MODEL> <PATH-TO-YOUR-DATASET>
```

Like before, after training you'll need to convert your PyTorch model to ONNX:

```bash
$ python onnx_export.py --model-dir=<YOUR-MODEL>
```

The converted model will be saved under `<YOUR-MODEL>/resnet18.onnx`, which you can then load with the `imagenet` programs like we did in the previous examples:

```bash
DATASET=<PATH-TO-YOUR-DATASET>

# C++ (MIPI CSI)
imagenet --model=<YOUR-MODEL>/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0

# Python (MIPI CSI)
imagenet.py --model=<YOUR-MODEL>/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0
```

If you need to, go back and collect more training data and re-train your model again.  You can restart the again and pick up where you left off using the `--resume` and `--epoch-start` flags (run `python train.py --help` for more info).  Remember to re-export the model to ONNX after re-training.


## What's Next

This is the last step of the *Hello AI World* tutorial, which covers inferencing and transfer learning on Jetson with TensorRT and PyTorch.  To recap, together we've covered:

* Using image recognition networks to classify images
* Coding your own image recognition programs in Python and C++
* Classifying video from a live camera stream
* Performing object detection to locate object coordinates
* Re-training models with PyTorch using transfer learning
* Collecting your own datasets and training your own models

Next we encourage you to experiment and apply what you've learned to other projects, perhaps taking advantage of Jetson's embedded form-factor - for example an autonomous robot or intelligent camera-based system.  Here are some example ideas that you could play around with:

* use GPIO to trigger external actuators or LEDs when an object is detected
* an autonomous robot that can find or follow an object
* a handheld battery-powered camera + Jetson + mini-display 
* an interactive toy or treat dispenser for your pet
* a smart doorbell camera that greets your guests

For more examples to inspire your creativity, see the **[Jetson Projects](https://developer.nvidia.com/embedded/community/jetson-projects)** page.  Have fun and good luck!

You can also follow our **[Two Days to a Demo](https://github.com/dusty-nv/jetson-inference#two-days-to-a-demo-DIGITS)** tutorial, which covers training of even larger datasets in the cloud or on a PC using discrete NVIDIA GPU(s).  Two Days to a Demo also covers semantic segmentation, which is like image classification, but on a per-pixel level instead of predicting one class for the entire image.


<p align="right">Next | <b><a href="pytorch-ssd.md">Re-training SSD-Mobilenet</a></b>
<br/>
Back | <b><a href="pytorch-plants.md">Re-training on the PlantCLEF Dataset</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
