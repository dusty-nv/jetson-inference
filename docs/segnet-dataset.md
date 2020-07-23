<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-camera.md">Back</a> | <a href="segnet-pretrained.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Semantic Segmentation with SegNet

The third deep learning capability we're highlighting in this tutorial is semantic segmentation.  Semantic segmentation is based on image recognition, except the classifications occur at the pixel level as opposed to classifying entire images as with image recognition.  This is accomplished by *convolutionalizing* a pre-trained image recognition model (like Alexnet), which turns it into a fully-convolutional segmentation model capable of per-pixel labelling.  Useful for environmental sensing and collision avoidance, segmentation yields dense per-pixel classification of many different potential objects per scene, including scene foregrounds and backgrounds.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-cityscapes.jpg)

The [`segNet`](../c/segNet.h) object accepts as input the 2D image, and outputs a second image with the per-pixel classification mask overlay.  Each pixel of the mask corresponds to the class of object that was classified.

> **note**:  see the DIGITS [semantic segmentation](https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation) example for more background info on segmentation.

### Downloading Aerial Drone Dataset

As an example of image segmentation, we'll work with an aerial drone dataset that separates ground terrain from the sky.  The dataset is in First Person View (FPV) to emulate the vantage point of a drone in flight and train a network that functions as an autopilot guided by the terrain that it senses.

To download and extract the dataset, run the following commands from the host PC running the DIGITS server:

``` bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/ft9cc5yjvrbhkh07wcivu5ji9zola6i1.gz -O NVIDIA-Aerial-Drone-Dataset.tar.gz

HTTP request sent, awaiting response... 200 OK
Length: 7140413391 (6.6G) [application/octet-stream]
Saving to: ‘NVIDIA-Aerial-Drone-Dataset.tar.gz’

NVIDIA-Aerial-Drone-Datase 100%[======================================>]   6.65G  3.33MB/s    in 44m 44s 

2017-04-17 14:11:54 (2.54 MB/s) - ‘NVIDIA-Aerial-Drone-Dataset.tar.gz’ saved [7140413391/7140413391]

$ tar -xzvf NVIDIA-Aerial-Drone-Dataset.tar.gz 
```

The dataset includes various clips captured from flights of drone platforms, but the one we'll be focusing on in this tutorial is under `FPV/SFWA`.  Next we'll create the training database in DIGITS before training the model.

### Importing the Aerial Dataset into DIGITS

First, navigate your browser to your DIGITS server instance and choose to create a new `Segmentation Dataset` from the drop-down in the Datasets tab:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-create-dataset.png" width="250">

In the dataset creation form, specify the following options and paths to the image and label folders under the location where you extracted the aerial dataset:

* Feature image folder:  `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/720p/images`
* Label image folder:   `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/720p/labels`
* set `% for validation` to 1%
* Class labels:  `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/fpv-labels.txt`
* Color map:  From text file
* Feature Encoding:  `None`
* Label Encoding:  `None`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-dataset-options.png)

Name the dataset whatever you choose and click the `Create` button at the bottom of the page to launch the importing job.  Next we'll create the new segmentation model and begin training.

##
<p align="right">Next | <b><a href="segnet-pretrained.md">Generating Pretrained FCN-Alexnet</a></b>
<br/>
Back | <b><a href="detectnet-camera.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
