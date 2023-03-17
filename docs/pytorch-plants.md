<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-cat-dog.md">Back</a> | <a href="pytorch-collect.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Classification</sup></s></p>

# Re-training on the PlantCLEF Dataset

Next, we'll train a model capable of classifying 20 different varieties of plants and trees from the <a href="https://www.imageclef.org/lifeclef/2017/plant">PlantCLEF</a> dataset.

<a href="https://www.imageclef.org/lifeclef/2017/plant"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants.jpg"></a>

Provided below is a 1.5GB subset that includes 10,475 training images, 1,155 validation images, and 30 test images across 20 classes of plants and trees.  The classes were selected from PlantCLEF 2017 from categories that had at least 500 training images in the original dataset:

```
• ash
• beech
• cat-tail
• cedar
• clover
• cyprus
• daisy
• dandelion
• dogwood
• elm
• fern
• fig
• fir
• juniper
• maple
• poison ivy
• sweetgum
• sycamore
• trout lily
• tulip tree
```

To get started, first make sure that you have [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) on your Jetson, then download the dataset below and kick off the training script.  After that, we'll test the re-trained model in TensorRT on some static images and a live camera feed. 

## Downloading the Data

Run these commands to download and extract the prepared PlantCLEF dataset:

``` bash
$ cd jetson-inference/python/training/classification/data
$ wget https://nvidia.box.com/shared/static/vbsywpw5iqy7r38j78xs0ctalg7jrg79.gz -O PlantCLEF_Subset.tar.gz
$ tar xvzf PlantCLEF_Subset.tar.gz
```

Mirrors of the dataset are available here:

* <a href="https://drive.google.com/file/d/14pUv-ZLHtRR-zCYjznr78mytFcnuR_1D/view?usp=sharing">https://drive.google.com/file/d/14pUv-ZLHtRR-zCYjznr78mytFcnuR_1D/view?usp=sharing</a>
* <a href="https://nvidia.box.com/s/vbsywpw5iqy7r38j78xs0ctalg7jrg79">https://nvidia.box.com/s/vbsywpw5iqy7r38j78xs0ctalg7jrg79</a>

## Re-training ResNet-18 Model

We'll use the same training script that we did from the previous example, located under <a href="https://github.com/dusty-nv/jetson-inference/tree/master/python/training/classification">`python/training/classification/`</a>.  By default it's set to train a ResNet-18 model, but you can change that with the `--arch` flag.

To launch the training, run the following commands:

``` bash
$ cd jetson-inference/python/training/classification
$ python3 train.py --model-dir=models/plants data/PlantCLEF_Subset
```

> **note:** if you run out of memory or your process is "killed" during training, try [Mounting SWAP](pytorch-transfer-learning.md#mounting-swap) and [Disabling the Desktop GUI](pytorch-transfer-learning.md#disabling-the-desktop-gui). <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to save memory, you can also reduce the `--batch-size` (default 8) and `--workers` (default 2)
  
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

<p align="center"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-training.jpg" width="700"></p>

At around epoch 30, the ResNet-18 model reaches 75% Top-5 accuracy, and at epoch 65 it converges on 85% Top-5 accuracy.  Interestingly these points of stability and convergence for the model occur at similiar times for ResNet-18 that they did for the previous Cat/Dog model.  The model's Top-1 accuracy is 55%, which we'll find to be quite effective in practice, given the diversity and challenging content from the PlantCLEF dataset (i.e. multiple overlapping varieties of plants per image and many pictures of leaves and tree trunks that are virtually indistinguishable from one another).  

By default the training script is set to run for 35 epochs, but if you don't wish to wait that long to test out your model, you can exit training early and proceed to the next step (optionally re-starting the training again later from where you left off).  You can also download this completed model that was trained for a full 100 epochs from here:

* <a href="https://nvidia.box.com/s/dslt9b0hqq7u71o6mzvy07w0onn0tw66">https://nvidia.box.com/s/dslt9b0hqq7u71o6mzvy07w0onn0tw66</a>

Note that the models are saved under `jetson-inference/python/training/classification/data/plants/`, including a checkpoint from the latest epoch and the best-performing model that has the highest classification accuracy.  You can change the directory that the models are saved to by altering the `--model-dir` flag.

## Converting the Model to ONNX

Just like with the Cat/Dog example, next we need to convert our trained model from PyTorch to ONNX, so that we can load it with TensorRT:

``` bash
python3 onnx_export.py --model-dir=models/plants
```

This will create a model called `resnet18.onnx` under `jetson-inference/python/training/classification/models/plants/`

## Processing Images with TensorRT

To classify some static test images, like before we'll use the extended command-line parameters to `imagenet` to load our customized ResNet-18 model that we re-trained above.  To run these commands, the working directory of your terminal should still be located in:  `jetson-inference/python/training/classification/`

```bash
NET=models/plants
DATASET=data/PlantCLEF_Subset

# C++
imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cattail.jpg cattail.jpg

# Python
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/cattail.jpg cattail.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-cattail.jpg" width="500">

```bash
# C++
imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/elm.jpg elm.jpg

# Python
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/elm.jpg elm.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-elm.jpg" width="500">

```bash
# C++
imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/juniper.jpg juniper.jpg

# Python
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/juniper.jpg juniper.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-juniper.jpg" width="500">

There are a bunch of test images included with the dataset, or you can download your own pictures to try.

### Processing all the Test Images

If you want to classify all of the test images at once, you can run the program on the entire directory:

``` bash
mkdir $DATASET/test_output

imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/../labels.txt \
           $DATASET/test $DATASET/test_output
```

In this instance, all the images will be read from the dataset's `test/` directory, and saved to the `test_output/` directory.  

For more info about loading/saving sequences of images, see the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page.

## Running the Live Camera Program

You can also try running your re-trained plant model on a live camera stream like below:

```bash
# C++ (MIPI CSI)
imagenet --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0

# Python (MIPI CSI)
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt csi://0
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-fern.jpg" width="500">

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-plants-poison-ivy.jpg" width="500">

Looks like I should be watching out for poison ivy!  

Next, we're going to cover a camera-based tool for collecting and labeling your own datasets captured from live video.  

<p align="right">Next | <b><a href="pytorch-collect.md">Collecting your own Classification Datasets</a></b>
<br/>
Back | <b><a href="pytorch-cat-dog.md">Re-training on the Cat/Dog Dataset</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
