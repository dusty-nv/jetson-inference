<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-custom.md">Back</a> | <a href="detectnet-snapshot.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></p> 

# Locating Objects with DetectNet
The previous image recognition examples output class probabilities representing the entire input image.   The second deep learning capability we're highlighting in this tutorial is detecting objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes).  This is performed using a 'detectNet' - or object detection / localization network.

The [`detectNet`](../c/detectNet.h) object accepts as input the 2D image, and outputs a list of coordinates of the detected bounding boxes.  To train the object detection model, first a pretrained ImageNet recognition model (like Googlenet) is used with bounding coordinate labels included in the training dataset in addition to the source imagery.

The following pretrained DetectNet models are included with the tutorial:

1. **ped-100**  (single-class pedestrian detector)
2. **multiped-500**   (multi-class pedestrian + baggage detector)
3. **facenet-120**  (single-class facial recognition detector)
4. **coco-airplane**  (MS COCO airplane class)
5. **coco-bottle**    (MS COCO bottle class)
6. **coco-chair**     (MS COCO chair class)
7. **coco-dog**       (MS COCO dog class)

As with the previous examples, provided are a console program and a camera streaming program for using detectNet.


### Detection Data Formatting in DIGITS

Example object detection datasets with include [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php), [MS-COCO](http://mscoco.org/), and others.  To use the KITTI dataset follow this [DIGITS object detection tutorial with KITTI](https://github.com/NVIDIA/DIGITS/blob/digits-4.0/digits/extensions/data/objectDetection/README.md).

Regardless of dataset, DIGITS uses KITTI metadata format for ingesting the detection bounding labels.  These consist of text files with frame numbers corresponding to image filenames, including contents such as:

```
dog 0 0 0 528.63 315.22 569.09 354.18 0 0 0 0 0 0 0
sheep 0 0 0 235.28 300.59 270.52 346.55 0 0 0 0 0 0 0
```

[Read more](https://github.com/NVIDIA/DIGITS/blob/digits-4.0/digits/extensions/data/objectDetection/README.md) about the folder structure and KITTI label format that DIGITS uses.  

### Downloading the Detection Dataset

Let's explore using the [MS-COCO](http://mscoco.org/) dataset to train and deploy networks that detect the locations of everyday objects in camera feeds.  See the [`coco2kitti.py`](../tools/coco2kitti.py) script for converting MS-COCO object classes to KITTI format.  Once in DIGITS folder structure, they can be imported as datasets into DIGITS.  Some example classes from MS-COCO already preprocessed in DIGITS/KITTI format are provided for convienience.

From a terminal on your DIGITS server download and extract **[sample MS-COCO classes](https://nvidia.box.com/shared/static/tdrvaw3fd2cwst2zu2jsi0u43vzk8ecu.gz)** already in DIGITS/KITTI format here:

```bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/tdrvaw3fd2cwst2zu2jsi0u43vzk8ecu.gz -O coco.tar.gz

HTTP request sent, awaiting response... 200 OK
Length: 5140413391 (4.5G) [application/octet-stream]
Saving to: ‘coco.tar.gz’

coco 100%[======================================>]   4.5G  3.33MB/s    in 28m 22s 

2017-04-17 10:41:19 (2.5 MB/s) - ‘coco.tar.gz’ saved [5140413391/5140413391]

$ tar -xzvf coco.tar.gz 
```

Included is the training data in DIGITS format for the airplane, bottle, chair, and dog classes.  [`coco2kitti.py`](../tools/coco2kitti.py) can be used to convert other classes.

### Importing the Detection Dataset into DIGITS

Navigate your browser to your DIGITS server instance and choose to create a new `Detection Dataset` from the drop-down in the Datasets tab:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-dataset-menu.png" width="250">

In the form fields, specify the following options and paths to the image and label folders under the location where you extracted the aerial dataset:

* Training image folder:  `coco/train/images/dog`
* Training label folder:  `coco/train/labels/dog`
* Validation image folder:  `coco/val/images/dog`
* Validation label folder:  `coco/val/labels/dog`
* Pad image (Width x Height):  `640 x 640`
* Custom classes:  `dontcare, dog`
* Group Name:  `MS-COCO`
* Dataset Name:  `coco-dog`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-dataset-dog.png)

Name the dataset whatever you choose and click the `Create` button at the bottom of the page to launch the importing job.  Next we'll create the new detection model and begin training it.

### Creating DetectNet Model with DIGITS

When the previous data import job is complete, return to the DIGITS home screen.  Select the `Models` tab and choose to create a new `Detection Model` from the drop-down:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-model-menu.png" width="250">

Make the following settings in the form:

* Select Dataset:  `coco-dog`
* Training epochs:  `100`
* Subtract Mean:  `none`
* Solver Type:  `Adam`
* Base learning rate:  `2.5e-05`
* Select `Show advanced learning options`
  * Policy:  `Exponential Decay`
  * Gamma:  `0.99`

#### Selecting DetectNet Batch Size

DetectNet's network default batch size of 10 consumes up to 12GB GPU memory during training.  However by using the `Batch Accumulation` field, you can also train DetectNet on a GPU with less than 12GB memory.  See the table below depending on the amount of GPU memory available in your DIGITS server:

| GPU Memory     | Batch Size                | Batch Accumulation  |
| -------------- |:-------------------------:|:-------------------:|
| 4GB            | 2                         | 5                   |
| 8GB            | 5                         | 2                   |
| 12GB or larger | `[network defaults]` (10) | Leave blank (1)     |

If you're training on a card with 12GB of memory or more, leave the `Batch Size` as the default and leave the `Batch Accumulation` blank.  For GPUs with less memory, use the settings from above.

#### Specifying the DetectNet Prototxt 

In the network area select the `Custom Network` tab and then copy/paste the contents of [`detectnet.prototxt`](../data/networks/detectnet.prototxt)

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-custom-network.jpg)

The DetectNet prototxt is located at [`data/networks/detectnet.prototxt`](https://github.com/dusty-nv/jetson-inference/blob/master/data/networks/detectnet.prototxt) in the repo.

#### Training the Model with Pretrained Googlenet

Since DetectNet is derived from Googlenet, it is strongly recommended to use pre-trained weights from Googlenet as this will help speed up and stabilize training significantly.  Download the Googlenet model from [here](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) or by running the following command from your DIGITS server:

```bash
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
```

Then specify the path to your Googlenet under the `Pretrained Model` field.

Select a GPU to train on and set a name and group for the model:

* Group Name `MS-COCO`
* Model Name `DetectNet-COCO-Dog`

Finally, click the `Create` button at the bottom to begin training.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-model-dog.png)

### Testing DetectNet Model Inference in DIGITS

Leave the training job to run for a while, say 50 epochs, until the mAP (`Mean Average Precision`) plot begins to increase.  Note that due to the way mAP is calculated by the DetectNet loss function, the scale of mAP isn't necessarily 0-100, and even an mAP between 5 and 10 may indicate the model is functional.  With the size of the example COCO datasets we are using, it should take a couple hours training on a recent GPU before all 100 epochs are complete.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-model-dog.png)

At this point, we can try testing our new model's inference on some example images in DIGITS.  On the same page as the plot above, scroll down under the `Trained Models` section.  Set the `Visualization Model` to *Bounding Boxes* and under `Test a Single Image`, select an image to try (for example, `/coco/val/images/dog/000074.png`):

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-visualization-options-dog.png" width="350">

Press the `Test One` button and you should see a page similar to:

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-infer-dog.png)

Next, we will deploy our DetectNet model to Jetson and run the inference there.

##
<p align="right">Next | <b><a href="detectnet-snapshot.md">Downloading the Detection Model to Jetson</a></b>
<br/>
Back | <b><a href="imagenet-custom.md">Loading Custom Models on Jetson</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
