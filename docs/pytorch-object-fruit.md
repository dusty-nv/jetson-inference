<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="../README.md#hello-ai-world">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Object Detection</sup></s></p>

# Re-training SSD-Mobilenet

Next, we'll train our own SSD-Mobilenet object detection model using PyTorch and the [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F06l9r) dataset.  SSD-Mobilenet is a popular network architecture for realtime object detection on mobile and embedded devices that combines the [SSD-300](https://arxiv.org/abs/1512.02325) Single-Shot MultiBox Detector with a [Mobilenet](https://arxiv.org/abs/1704.04861) backbone.  

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-ssd-mobilenet.jpg">

In the example below, we'll train a custom detection model that locates 8 different varieties of fruit, although you are welcome to pick from any of the [600 classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) in the Open Images dataset to train your model on.

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-fruit.jpg">

To get started, first make sure that you have [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3.6** on your Jetson, then download the base model from below and install some packages that are needed for training the SSD model.  Then after downloading your desired object classes, you will kick off the training script (and probably let it run overnight).  After that, we'll test the re-trained detection model in TensorRT on some static images and a live camera feed. 

## Setup

> **note:** first make sure that you have [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3.6** on your Jetson  

The PyTorch code for training SSD-Mobilenet is found in the repo under [`jetson-inference/python/training/detection/ssd`](https://github.com/dusty-nv/pytorch-ssd).  There are a couple steps required before using it:

```bash
$ cd jetson-inference/python/training/detection/ssd
$ wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
$ pip3 install -v -r requirements.txt
```

This will download the [base model](https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth) to `ssd/models` and install some required Python packages.  The base model was already pre-trained on a different dataset (PASCAL VOC) so that we don't need to train SSD-Mobilenet from scratch, which would take much longer.  Instead we'll use transfer learning to fine-tune it to detect new object classes of our choosing.


## Downloading the Data

The [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0fp6w) dataset contains over [600 object classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) that you can pick and choose from.  There is a script provided called `open_images_downloader.py` which will automatically download the desired object classes for you.  The example classes that we'll be using are `"Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"` (although you can substitute your own choices selected from the [class list](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt)). 

> **note:**  the fewer classes used, the faster the model will run during inferencing
> &nbsp;&nbsp;&nbsp;&nbsp; before downloading your own classes, see [Limiting the Amount of Data](#limiting-the-amount-of-data) below.

```bash
$ python3 open_images_downloader.py --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"

2020-07-08 17:02:39,877 - root - Starting to download 6750 images.
2020-07-08 17:02:49,710 - root - Downloaded 100 images.
2020-07-08 17:02:54,382 - root - Downloaded 200 images.
2020-07-08 17:03:00,127 - root - Downloaded 300 images.
2020-07-08 17:03:04,542 - root - Downloaded 400 images.
2020-07-08 17:03:09,346 - root - Downloaded 500 images.
...
2020-07-08 17:45:30,192 - root - Task Done.
```

By default, the dataset will be download to the `ssd/data` directory, but you can change that by specifying the `--root <PATH>` option.  Depending on the size of your dataset, it may be necessary to use external storage.

### Limiting the Amount of Data

Depending on the classes that you select, Open Images can contain lots of data - in some cases too much to be feasibly trained on a Jetson.  In particular the classes containing people and vehicles have an abundance of images.  

So when selecting your own classes, before downloading the data it's recommended to first run the downloader script with the `--stats-only` option.  This will show how many bounding box instances and images there are in your classes, without actually downloading any images.  

``` bash
$ python3 open_images_downloader.py --stats-only --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"
...
-------------------------------------
 overall bounding box counts
-------------------------------------
  train set:      24863
  validation set: 853
  test set:       2874

total bounding boxes: 28590
total images:         6750
```

> **note:** `--stats-only` does download the bounding box annotation data (approximately ~1GB), but not the images yet.  

In practice, to keep the training time down (and disk space), you probably want to keep the total number of bounding boxes <50K.  You can limit the amount of data downloaded with the `--max-boxes` option, which indirectly scales the number of images downloaded (trying to directly limit the number of images isn't easily done, as each image may have multiple bounding boxes from multiple classes, so instead we control the number of bounding annotations).

For example, if you wanted to only download 15K boxes for the fruit dataset, you would launch the image download like this instead:

``` bash
$ python3 open_images_downloader.py --max-boxes=15000 --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"
```

If `--max-boxes` isn't set, by default all the data available will be downloaded - so be sure to check the amount of data with `--stats-only` first before downloading.  Setting `--max-boxes` will split the annotation allocations between 80% train, 10% val, and 10% test sets.


Mirrors of the dataset are available here:

* <a href="https://drive.google.com/file/d/14pUv-ZLHtRR-zCYjznr78mytFcnuR_1D/view?usp=sharing">https://drive.google.com/file/d/14pUv-ZLHtRR-zCYjznr78mytFcnuR_1D/view?usp=sharing</a>
* <a href="https://nvidia.box.com/s/vbsywpw5iqy7r38j78xs0ctalg7jrg79">https://nvidia.box.com/s/vbsywpw5iqy7r38j78xs0ctalg7jrg79</a>

## Re-training ResNet-18 Model

We'll use the same training script that we did from the previous example, located under <a href="https://github.com/dusty-nv/jetson-inference/tree/master/python/training/classification">`python/training/classification/`</a>.  By default it's set to train a ResNet-18 model, but you can change that with the `--arch` flag.

To launch the training, run the following commands:

``` bash
$ cd jetson-inference/python/training/classification
$ python train.py --model-dir=plants ~/datasets/PlantCLEF_Subset
```

As training begins, you should see text from the console like the following:

``` bash
Use GPU: 0 for training
=> dataset classes:  20 ['ash', 'beech', 'cattail', 'cedar', 'clover', 'cyprus', 'daisy', 'dandelion', 'dogwood', 'elm', 'fern', 'fig', 'fir', 'juniper', 'maple', 'poison_ivy', 'sweetgum', 'sycamore', 'trout_lily', 'tulip_tree']
=> using pre-trained model 'resnet18'
=> reshaped ResNet fully-connected layer with: Linear(in_features=512, out_features=20, bias=True)
Epoch: [0][   0/1307]	Time 49.345 (49.345)	Data  0.561 ( 0.561)	Loss 3.2172e+00 (3.2172e+00)	Acc@1   0.00 (  0.00)	Acc@5  25.00 ( 25.00)
Epoch: [0][  10/1307]	Time  0.779 ( 5.211)	Data  0.000 ( 0.060)	Loss 2.3915e+01 (1.5221e+01)	Acc@1   0.00 (  5.68)	Acc@5  12.50 ( 27.27)
Epoch: [0][  20/1307]	Time  0.765 ( 3.096)	Data  0.000 ( 0.053)	Loss 3.6293e+01 (2.1256e+01)	Acc@1   0.00 (  5.95)	Acc@5  37.50 ( 27.38)
Epoch: [0][  30/1307]	Time  0.773 ( 2.346)	Data  0.000 ( 0.051)	Loss 2.8803e+00 (1.9256e+01)	Acc@1  37.50 (  6.85)	Acc@5  62.50 ( 27.42)
Epoch: [0][  40/1307]	Time  0.774 ( 1.962)	Data  0.000 ( 0.050)	Loss 3.7734e+00 (1.5865e+01)	Acc@1  12.50 (  8.84)	Acc@5  37.50 ( 29.88)
Epoch: [0][  50/1307]	Time  0.772 ( 1.731)	Data  0.000 ( 0.049)	Loss 3.0311e+00 (1.3756e+01)	Acc@1  25.00 ( 10.29)	Acc@5  37.50 ( 32.35)
Epoch: [0][  60/1307]	Time  0.773 ( 1.574)	Data  0.000 ( 0.048)	Loss 3.2433e+00 (1.2093e+01)	Acc@1   0.00 (  9.84)	Acc@5  25.00 ( 32.79)
Epoch: [0][  70/1307]	Time  0.806 ( 1.462)	Data  0.000 ( 0.048)	Loss 2.9213e+00 (1.0843e+01)	Acc@1  12.50 (  8.98)	Acc@5  37.50 ( 33.27)
Epoch: [0][  80/1307]	Time  0.792 ( 1.379)	Data  0.000 ( 0.048)	Loss 3.2370e+00 (9.8715e+00)	Acc@1   0.00 (  9.26)	Acc@5  25.00 ( 34.41)
Epoch: [0][  90/1307]	Time  0.770 ( 1.314)	Data  0.000 ( 0.048)	Loss 2.4494e+00 (9.0905e+00)	Acc@1  25.00 (  9.75)	Acc@5  75.00 ( 36.26)
Epoch: [0][ 100/1307]	Time  0.801 ( 1.261)	Data  0.001 ( 0.048)	Loss 2.6449e+00 (8.4769e+00)	Acc@1  25.00 ( 10.40)	Acc@5  62.50 ( 37.00)
```

See the [Training Metrics](pytorch-cat-dog.md#training-metrics) from the previous page for a description of the statistics from the output above.

### Model Accuracy

On the PlantCLEF dataset of 10,475 images, training ResNet-18 takes approximately ~15 minutes per epoch on Jetson Nano, or around 8 hours to train the model for 35 epochs.  Below is a graph for analyzing the training progression of epochs versus model accuracy:

<p align="center"><img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-training.jpg" width="700"></p>

At around epoch 30, the ResNet-18 model reaches 75% Top-5 accuracy, and at epoch 65 it converges on 85% Top-5 accuracy.  Interestingly these points of stability and convergence for the model occur at similiar times for ResNet-18 that they did for the previous Cat/Dog model.  The model's Top-1 accuracy is 55%, which we'll find to be quite effective in practice, given the diversity and challenging content from the PlantCLEF dataset (i.e. multiple overlapping varieties of plants per image and many pictures of leaves and tree trunks that are virtually indistinguishable from one another).  

By default the training script is set to run for 35 epochs, but if you don't wish to wait that long to test out your model, you can exit training early and proceed to the next step (optionally re-starting the training again later from where you left off).  You can also download this completed model that was trained for a full 100 epochs from here:

* <a href="https://nvidia.box.com/s/dslt9b0hqq7u71o6mzvy07w0onn0tw66">https://nvidia.box.com/s/dslt9b0hqq7u71o6mzvy07w0onn0tw66</a>

Note that the models are saved under `jetson-inference/python/training/classification/plants/`, including a checkpoint from the latest epoch and the best-performing model that has the highest classification accuracy.  You can change the directory that the models are saved to by altering the `--model-dir` flag.

## Converting the Model to ONNX

Just like with the Cat/Dog example, next we need to convert our trained model from PyTorch to ONNX, so that we can load it with TensorRT:

``` bash
python onnx_export.py --model-dir=plants
```

This will create a model called `resnet18.onnx` under `jetson-inference/python/training/classification/plants/`

## Processing Images with TensorRT

To classify some static test images, like before we'll use the extended command-line parameters to `imagenet-console` to load our customized ResNet-18 model that we re-trained above.  To run these commands, the working directory of your terminal should still be located in:  `jetson-inference/python/training/classification/`

```bash
DATASET=~/datasets/PlantCLEF_Subset

# C++
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cattail.jpg cattail.jpg

# Python
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cattail.jpg cattail.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-cattail.jpg" width="500">

```bash
# C++
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/elm.jpg elm.jpg

# Python
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/elm.jpg elm.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-elm.jpg" width="500">

```bash
# C++
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/juniper.jpg juniper.jpg

# Python
imagenet-console --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/juniper.jpg juniper.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-juniper.jpg" width="500">

There are a bunch of test images included with the dataset, or you can download your own pictures to try.

### Processing all the Test Images

If you want to classify all of the test images without having to do them individually, you can create a simple script like below that loops over them and outputs to the `test_output` directory under the dataset:

```bash
#!/bin/bash  
NET="~/jetson-inference/python/training/classification/plants"
DATASET="~/datasets/PlantCLEF_Subset"

cd $DATASET
cp -r test test_output

FILES="$DATASET/test_output/*.jpg"

cd $DATA

for f in $FILES
do
     echo "Processing $f"
     imagenet-console --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt $f $f
done
```

## Running the Live Camera Program

You can also try running your re-trained plant model on a live camera stream like below:

```bash
DATASET=~/datasets/PlantCLEF_Subset

# C++
imagenet-camera --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt

# Python
imagenet-camera.py --model=plants/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-fern.jpg" width="500">

<img src="https://github.com/dusty-nv/jetson-inference/raw/python/docs/images/pytorch-plants-poison-ivy.jpg" width="500">

Looks like I should be watching out for poison ivy!  

Next, we're going to cover a camera-based tool for collecting and labelling your own datasets captured from live video.  

<p align="right">Next | <b><a href="pytorch-collect.md">Collecting your own Datasets</a></b>
<br/>
Back | <b><a href="pytorch-plants.md">Re-training on the Cat/Dog Dataset</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
