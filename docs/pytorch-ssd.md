<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-collect.md">Back</a> | <a href="pytorch-collect-detection.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Object Detection</sup></s></p>

# Re-training SSD-Mobilenet

Next, we'll train our own SSD-Mobilenet object detection model using PyTorch and the [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F06l9r) dataset.  SSD-Mobilenet is a popular network architecture for realtime object detection on mobile and embedded devices that combines the [SSD-300](https://arxiv.org/abs/1512.02325) Single-Shot MultiBox Detector with a [Mobilenet](https://arxiv.org/abs/1704.04861) backbone.  

<a href="https://arxiv.org/abs/1512.02325"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-ssd-mobilenet.jpg"></a>

In the example below, we'll train a custom detection model that locates 8 different varieties of fruit, although you are welcome to pick from any of the [600 classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) in the Open Images dataset to train your model on.  You can visually browse the dataset [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection).

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-fruit.jpg">

To get started, first make sure that you have [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) (or newer) and [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3** on your Jetson.  JetPack 4.4 includes TensorRT 7.1, which is the minimum TensorRT version that supports loading SSD-Mobilenet via ONNX.  Newer versions of TensorRT are fine too.

## Setup

The PyTorch code for training SSD-Mobilenet is found in the repo under [`jetson-inference/python/training/detection/ssd`](https://github.com/dusty-nv/pytorch-ssd).  If you aren't [Running the Docker Container](aux-docker.md), there are a couple steps required before using it:

```bash
# you only need to run these if you aren't using the container
$ cd jetson-inference/python/training/detection/ssd
$ wget https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O models/mobilenet-v1-ssd-mp-0_675.pth
$ pip3 install -v -r requirements.txt
```

> **note:** first make sure that you have [JetPack 4.4](https://developer.nvidia.com/embedded/jetpack) or newer on your Jetson and [PyTorch installed](pytorch-transfer-learning.md#installing-pytorch) for **Python 3**

This will download the [base model](https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth) to `ssd/models` and install some required Python packages (these were already installed into the container).  The base model was already pre-trained on a different dataset (PASCAL VOC) so that we don't need to train SSD-Mobilenet from scratch, which would take much longer.  Instead we'll use transfer learning to fine-tune it to detect new object classes of our choosing.

## Downloading the Data

The [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0fp6w) dataset contains over [600 object classes](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt) that you can pick and choose from.  There is a script provided called `open_images_downloader.py` which will automatically download the desired object classes for you.  

> **note:**  the fewer classes used, the faster the model will run during inferencing.  Open Images can also contain hundreds of gigabytes of data depending on the classes you pick - so before downloading your own classes, see the [Limiting the Amount of Data](#limiting-the-amount-of-data) section below.

The classes that we'll be using are `"Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon"`, for example for a fruit-picking robot - although you are welcome to substitute your own choices from the [class list](https://github.com/dusty-nv/pytorch-ssd/blob/master/open_images_classes.txt). The fruit classes have ~6500 images, which is a happy medium.

```bash
$ python3 open_images_downloader.py --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit
...
2020-07-09 16:20:42 - Starting to download 6360 images.
2020-07-09 16:20:42 - Downloaded 100 images.
2020-07-09 16:20:42 - Downloaded 200 images.
2020-07-09 16:20:42 - Downloaded 300 images.
2020-07-09 16:20:42 - Downloaded 400 images.
2020-07-09 16:20:42 - Downloaded 500 images.
2020-07-09 16:20:46 - Downloaded 600 images.
...
2020-07-09 16:32:12 - Task Done.
```

By default, the dataset will be downloaded to the `data/` directory under `jetson-inference/python/training/detection/ssd` (which is automatically [mounted into the container](aux-docker.md#mounted-data-volumes)), but you can change that by specifying the `--data=<PATH>` option.  Depending on the size of your dataset, it may be necessary to use external storage.  And if you download multiple datasets, you should store each in their own subdirectory.

### Limiting the Amount of Data

Depending on the classes that you select, Open Images can contain lots of data - in some cases too much to be trained in a reasonable amount of time for our purposes.  In particular, the classes containing people and vehicles have a very large amount of images (>250GB).  

So when selecting your own classes, before downloading the data it's recommended to first run the downloader script with the `--stats-only` option.  This will show how many images there are for your classes, without actually downloading any images.  

``` bash
$ python3 open_images_downloader.py --stats-only --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit
...
2020-07-09 16:18:06 - Total available images: 6360
2020-07-09 16:18:06 - Total available boxes:  27188

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

In practice, to keep the training time down (and disk space), you probably want to keep the total number of images <10K.  Although the more images you use, the more accurate your model will be.  You can limit the amount of data downloaded with the `--max-images` option or the `--max-annotations-per-class` options:

* `--max-images` limits the total dataset to the specified number of images, while keeping the distribution of images per class roughly the same as the original dataset.  If one class has more images than another, the ratio will remain roughly the same. 
* `--max-annotations-per-class` limits each class to the specified number of bounding boxes, and if a class has less than that number available, all of it's data will be used - this is useful if the distribution of data is unbalanced across classes.

For example, if you wanted to only use 2500 images for the fruit dataset, you would launch the downloader like this:

``` bash
$ python3 open_images_downloader.py --max-images=2500 --class-names "Apple,Orange,Banana,Strawberry,Grape,Pear,Pineapple,Watermelon" --data=data/fruit
```

If the `--max-boxes` option or `--max-annotations-per-class` isn't set, by default all the data available will be downloaded - so beforehand, be sure to check the amount of data first with `--stats-only`.  Unfortunately it isn't possible in advance to determine the actual disk size requirements of the images, but a general rule of thumb for this dataset is to budget ~350KB per image (~2GB for the fruits).

### Training Performance

Below is approximate SSD-Mobilenet training performance to help estimate the time required for training:

|           | Images/sec | Time per epoch* |
|-----------|:----------:|:---------------:|
| Nano      |    4.77    |  17 min 55 sec  |
| Xavier NX |    14.65   |   5 min 50 sec  |

* measured on the fruits dataset (5145 training images, batch size 4)

## Training the SSD-Mobilenet Model

Once your data has finished downloading, run the `train_ssd.py` script to launch the training:

```bash
python3 train_ssd.py --data=data/fruit --model-dir=models/fruit --batch-size=4 --epochs=30
```

> **note:** if you run out of memory or your process is "killed" during training, try [Mounting SWAP](pytorch-transfer-learning.md#mounting-swap) and [Disabling the Desktop GUI](pytorch-transfer-learning.md#disabling-the-desktop-gui). <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to save memory, you can also reduce the `--batch-size` (default 4) and `--workers` (default 2)
  
Here are some common options that you can run the training script with:

| Argument       |  Default  | Description                                                |
|----------------|:---------:|------------------------------------------------------------|
| `--data`       |  `data/`  | the location of the dataset                                |
| `--model-dir`  | `models/` | directory to output the trained model checkpoints          |
| `--resume`     |    None   | path to an existing checkpoint to resume training from     |
| `--batch-size` |     4     | try increasing depending on available memory               |
| `--epochs`     |     30    | up to 100 is desirable, but will increase training time    |
| `--workers`    |     2     | number of data loader threads (0 = disable multithreading) |

Over time, you should see the loss decreasing:

```bash
2020-07-10 13:14:12 - Epoch: 0, Step: 10/1287, Avg Loss: 12.4240, Avg Regression Loss 3.5747, Avg Classification Loss: 8.8493
2020-07-10 13:14:12 - Epoch: 0, Step: 20/1287, Avg Loss: 9.6947, Avg Regression Loss 4.1911, Avg Classification Loss: 5.5036
2020-07-10 13:14:13 - Epoch: 0, Step: 30/1287, Avg Loss: 8.7409, Avg Regression Loss 3.4078, Avg Classification Loss: 5.3332
2020-07-10 13:14:13 - Epoch: 0, Step: 40/1287, Avg Loss: 7.3736, Avg Regression Loss 2.5356, Avg Classification Loss: 4.8379
2020-07-10 13:14:14 - Epoch: 0, Step: 50/1287, Avg Loss: 6.3461, Avg Regression Loss 2.2286, Avg Classification Loss: 4.1175
...
2020-07-10 13:19:26 - Epoch: 0, Validation Loss: 5.6730, Validation Regression Loss 1.7096, Validation Classification Loss: 3.9634
2020-07-10 13:19:26 - Saved model models/fruit/mb1-ssd-Epoch-0-Loss-5.672993580500285.pth
```

To test your model before the full number of epochs have completed training, you can press `Ctrl+C` to kill the training script, and resume it again later with the `--resume=<CHECKPOINT>` argument.  You can download the fruit model that was already trained for 100 epochs [here](https://nvidia.box.com/shared/static/gq0zlf0g2r258g3ldabl9o7vch18cxmi.gz).

## Converting the Model to ONNX

Next we need to convert our trained model from PyTorch to ONNX, so that we can load it with TensorRT:

``` bash
python3 onnx_export.py --model-dir=models/fruit
```

This will save a model called `ssd-mobilenet.onnx` under `jetson-inference/python/training/detection/ssd/models/fruit/`

## Processing Images with TensorRT

To classify some static test images, we'll use the extended command-line parameters to `detectnet` (or `detectnet.py`) to load our custom SSD-Mobilenet ONNX model.  To run these commands, the working directory of your terminal should still be located in:  `jetson-inference/python/training/detection/ssd/`

```bash
IMAGES=<path-to-your-jetson-inference>/data/images   # substitute your jetson-inference path here

detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            "$IMAGES/fruit_*.jpg" $IMAGES/test/fruit_%i.jpg
```

> **note:**  `detectnet.py` can be substituted above to run the Python version of the program

Below are some of the images output to the `$IMAGES/test` directory:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-fruit-2.jpg">

## Running the Live Camera Program

You can also try running your re-trained plant model on a camera or video stream like below:

```bash
detectnet --model=models/fruit/ssd-mobilenet.onnx --labels=models/fruit/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
```

For more details about other camera/video sources, please see [Camera Streaming and Multimedia](aux-streaming.md).

<p align="right">Next | <b><a href="pytorch-collect-detection.md">Collecting your own Detection Datasets</a></b>
<br/>
Back | <b><a href="pytorch-collect.md">Collecting your own Classification Datasets</a></p>
</b><p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
