<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="../README.md#hello-ai-world">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Object Detection</sup></s></p>

# Re-training SSD-Mobilenet

Next, we'll train our own SSD-Mobilenet object detection model using PyTorch and the [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F06l9r) dataset.  SSD-Mobilenet is a popular network architecture for realtime object detection on mobile and embedded devices that combines the [SSD-300](https://arxiv.org/abs/1512.02325) Single-Shot MultiBox Detector with a [Mobilenet](https://arxiv.org/abs/1704.04861) backbone.  

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-ssd-mobilenet.jpg">

In the example below, we'll train a custom detection model that locates 8 different varieties of fruit, although you are welcome to pick from any of the [600 classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) in the Open Images dataset to train your model on.  You can visually browse the dataset [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection).

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-fruit.jpg">

To get started, first make sure that you have [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) or newer and [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3.6** on your Jetson.  JetPack 4.4 includes TensorRT 7.1, which is the minimum TensorRT version that supports loading SSD-Mobilenet via ONNX.  And the PyTorch training scripts used for training SSD-Mobilenet are for Python3, so PyTorch should be installed for Python 3.6.

## Setup

> **note:** first make sure that you have [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) or newer on your Jetson and [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3.6**

The PyTorch code for training SSD-Mobilenet is found in the repo under [`jetson-inference/python/training/detection/ssd`](https://github.com/dusty-nv/pytorch-ssd).  There are a couple steps required before using it:

```bash
$ cd jetson-inference/python/training/detection/ssd
$ wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
$ pip3 install -v -r requirements.txt
```

This will download the [base model](https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth) to `ssd/models` and install some required Python packages.  The base model was already pre-trained on a different dataset (PASCAL VOC) so that we don't need to train SSD-Mobilenet from scratch, which would take much longer.  Instead we'll use transfer learning to fine-tune it to detect new object classes of our choosing.

## Downloading the Data

The [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0fp6w) dataset contains over [600 object classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) that you can pick and choose from.  There is a script provided called `open_images_downloader.py` which will automatically download the desired object classes for you.  

> **note:**  the fewer classes used, the faster the model will run during inferencing.  </br>
> &nbsp;&nbsp;&nbsp;&nbsp; before downloading your own classes, see [Limiting the Amount of Data](#limiting-the-amount-of-data) below.

The classes that we'll be using are `"Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"`, for example for a fruit-picking robot. Although you are welcome to substitute your own choices from the [class list](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt). 

```bash
$ python3 open_images_downloader.py --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"
...
2020-07-09 16:20:42,778 - Starting to download 6360 images.
2020-07-09 16:20:42,821 - Downloaded 100 images.
2020-07-09 16:20:42,833 - Downloaded 200 images.
2020-07-09 16:20:42,845 - Downloaded 300 images.
2020-07-09 16:20:42,862 - Downloaded 400 images.
2020-07-09 16:20:42,877 - Downloaded 500 images.
2020-07-09 16:20:46,494 - Downloaded 600 images.
...
2020-07-09 16:32:12,321 - Task Done.
```

By default, the dataset will be downloaded to the `data/` directory under `jetson-inference/python/training/detection/ssd`, but you can change that by specifying the `--data=<PATH>` option.  Depending on the size of your dataset, it may be necessary to use external storage.  And if you download multiple datasets, you should store each dataset in their own subdirectory.

### Limiting the Amount of Data

Depending on the classes that you select, Open Images can contain lots of data - in some cases too much to be trained in a reasonable amount of time for our purposes.  In particular, the classes containing people and vehicles have a very large amount of images (>250GB).  

So when selecting your own classes, before downloading the data it's recommended to first run the downloader script with the `--stats-only` option.  This will show how many images there are for your classes, without actually downloading any images.  

``` bash
$ python3 open_images_downloader.py --stats-only --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"
...
2020-07-09 16:18:06,879 - Total available images: 6360
2020-07-09 16:18:06,879 - Total available boxes:  27188

-------------------------------------
 'train' set statistics
-------------------------------------
  Image count:  5145
  Bounding box count:  23539
  Bounding box distribution:
    Strawberry:  7553/23539 = 0.32
    Orange:  6186/23539 = 0.26
    Apple:  3622/23539 = 0.15
    Grape:  2560/23539 = 0.11
    Banana:  1574/23539 = 0.07
    Pear:  757/23539 = 0.03
    Watermelon:  753/23539 = 0.03
    Pineapple:  534/23539 = 0.02

...

-------------------------------------
 Overall statistics
-------------------------------------
  Image count:  6360
  Bounding box count:  27188
```

> **note:** `--stats-only` does download the annotation data (approximately ~1GB), but not the images yet.  

In practice, to keep the training time down (and disk space), you probably want to keep the total number of images <10K.  Although the more images you use, the more accurate your model will be.  You can limit the amount of data downloaded with the `--max-images` option.

For example, if you wanted to only use 2500 images for the fruit dataset, you would launch the downloader like this:

``` bash
$ python3 open_images_downloader.py --max-images=2500 --class_names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"
```

If the `--max-boxes` option isn't set, by default all the data available will be downloaded - so be sure to check the amount of data first with `--stats-only`.  Unfortunately it isn't possible in advance to determine the actual disk size requirements of the images, but a general rule of thumb for this dataset is to budget ~350KB per image.


## Training the SSD-Mobilenet Model

Once your data has finished downloading, run the `train_ssd.py` script to launch the training:

```bash
python3 train_ssd.py --model-dir=models/fruit --batch-size=4 --num-epochs=30
```

Here are some common options that you can run the training script with:

| Argument       |  Default  | Description                                             |
|----------------|:---------:|---------------------------------------------------------|
| `--data`       |  `data/`  | the location of the dataset                             |
| `--model-dir`  | `models/` | directory to output the trained model checkpoints       |
| `--resume`     |    None   | path to an existing checkpoint to resume training from  |
| `--batch-size` |     4     | try increasing depending on available memory            |
| `--num-epochs` |     30    | up to 100 is desirable, but will increase training time |

Over time, you should see the loss decreasing:

```bash
2020-07-10 13:14:12,076 - Epoch: 0, Step: 10/1287, Avg Loss: 12.4240, Avg Regression Loss 3.5747, Avg Classification Loss: 8.8493
2020-07-10 13:14:12,688 - Epoch: 0, Step: 20/1287, Avg Loss: 9.6947, Avg Regression Loss 4.1911, Avg Classification Loss: 5.5036
2020-07-10 13:14:13,145 - Epoch: 0, Step: 30/1287, Avg Loss: 8.7409, Avg Regression Loss 3.4078, Avg Classification Loss: 5.3332
2020-07-10 13:14:13,688 - Epoch: 0, Step: 40/1287, Avg Loss: 7.3736, Avg Regression Loss 2.5356, Avg Classification Loss: 4.8379
2020-07-10 13:14:14,293 - Epoch: 0, Step: 50/1287, Avg Loss: 6.3461, Avg Regression Loss 2.2286, Avg Classification Loss: 4.1175
...
2020-07-10 13:19:26,971 - Epoch: 0, Validation Loss: 5.6730, Validation Regression Loss 1.7096, Validation Classification Loss: 3.9634
2020-07-10 13:19:26,997 - Saved model models/fruit/mb1-ssd-Epoch-0-Loss-5.672993580500285.pth
```

If you want to test your model before training for the full number of epochs, you can press `Ctrl+C` to kill the training script, and resume it again later on using the `--resume=<CHECKPOINT>` argument.

## Converting the Model to ONNX

Next we need to convert our trained model from PyTorch to ONNX, so that we can load it with TensorRT:

``` bash
python3 onnx_export.py --model-dir=models/fruit
```

This will create a model called `ssd-mobilenet.onnx` under `jetson-inference/python/training/detection/ssd/models/fruit/`

## Processing Images with TensorRT

To classify some static test images, we'll use the extended command-line parameters to `detectnet` (or `detectnet.py`) to load our custom SSD-Mobilenet ONNX model.  To run these commands, the working directory of your terminal should still be located in:  `jetson-inference/python/training/classification/`

```bash
mkdir test_fruit

detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
		"images/fruit_*.jpg" test_fruit
```

> **note:**  `detectnet.py` can be substituted above to run the Python version of the program

Below are some of the images output to the `test_fruit/` directory:

<img src="https://github.com/dusty-nv/jetson-inference/raw/dev/docs/images/pytorch-plants-2.jpg"

## Running the Live Camera Program

You can also try running your re-trained plant model on a camera or video stream like below:

```bash
detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
		csi://0
```

For more details about camera/video options, please see [Camera Streaming and Multimedia](aux-streaming.md).

<p align="right">Next | <b><a href="TODO">Collecting your own Detection Datasets (TODO)</a></b>
<br/>
Back | <b><a href="pytorch-transfer-learning.md">Transfer Learning with PyTorch</a></p>
</b><p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
