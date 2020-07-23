<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="detectnet-snapshot.md">Back</a> | <a href="detectnet-camera.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# Detecting Objects from the Command Line

To process test images with [`detectNet`](../c/detectNet.h) and TensorRT on the Jetson, we can use the [`detectnet-console`](../detectnet-console/detectnet-console.cpp) program.  

[`detectnet-console`](../detectnet-console/detectnet-console.cpp) accepts command-line arguments representing the path to the input image and path to the output image (with the bounding box overlays rendered).  Some test images are also included with the repo.

To specify your model that you downloaded from DIGITS in the previous step, use the syntax to `detectnet-console` below.  First, for convienience, set the path to your extracted snapshot into a `$NET` variable:

``` bash
$ NET=20170504-190602-879f_epoch_100

$ ./detectnet-console dog_0.jpg output_0.jpg \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_38600.caffemodel \
--input_blob=data \ 
--output_cvg=coverage \
--output_bbox=bboxes
```

> **note:**  the `input_blob`, `output_cvg`, and `output_bbox` arguments may be omitted if your DetectNet layer names match the defaults above (i.e. if you are using the prototxt from following this tutorial). These optional command line parameters are provided if you are using a customized DetectNet with different layer names.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-0.jpg)

### Launching With a Pretrained Model

Alternatively, to load one of the pretrained snapshots that comes with the repo, you can specify the optional `--network` flag which changes the detection model being used (the default network is PedNet).  

Here's an example of locating humans in an image with the default PedNet model:

#### C++

``` bash
$ ./detectnet-console peds-004.jpg output.jpg
```

#### Python

``` bash
$ ./detectnet-console.py peds-004.jpg output.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-00.jpg" width="900">


### Pre-trained Detection Models Available

Below is a table of the pre-trained object detection networks available for [download](building-repo.md#downloading-models), and the associated `--network` argument to `detectnet-console` used for loading the pre-trained models:

| Model                   | CLI argument       | NetworkType enum   | Object classes       |
| ------------------------|--------------------|--------------------|----------------------|
| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
| SSD-Inception-v2        | `ssd-inception-v2` | `SSD_INCEPTION_V2` | 91 ([COCO classes](https://raw.githubusercontent.com/AastaNV/TRT_object_detection/master/coco.py))     |
| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
| facenet-120             | `facenet`          | `FACENET`          | faces                |

> **note**:  to download additional networks, run the [Model Downloader](building-repo.md#downloading-models) tool<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ cd jetson-inference/tools` <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`$ ./download-models.sh` <br/>


### Running Different Detection Models

You can specify which model to load by setting the `--network` flag on the command line to one of the corresponding CLI arguments from the table above.  By default, PedNet is loaded (pedestrian detection) if the optional `--network` flag isn't specified.

Let's try running some of the other COCO models:

``` bash
# C++
$ ./detectnet-console --network=coco-dog dog_1.jpg output_1.jpg

# Python
$ ./detectnet-console.py --network=coco-dog dog_1.jpg output_1.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-1.jpg)

``` bash
# C++
$ ./detectnet-console --network=coco-bottle bottle_0.jpg output_2.jpg

# Python
$ ./detectnet-console.py --network=coco-bottle bottle_0.jpg output_2.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-bottle-0.jpg)

``` bash
# C++
$ ./detectnet-console --network=coco-airplane airplane_0.jpg output_3.jpg 

# Python
$ ./detectnet-console.py --network=coco-airplane airplane_0.jpg output_3.jpg
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-airplane-0.jpg)


### Multi-class Object Detection Models

Some models support the detection of multiple types of objects.  For example, when using the `multiped` model on images containing luggage or baggage in addition to pedestrians, the 2nd object class is rendered with a green overlay:

``` bash
# C++
$ ./detectnet-console --network=multiped peds-003.jpg output_4.jpg

# Python
$ ./detectnet-console.py --network=multiped peds-003.jpg output_4.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-01.jpg" width="900">

Next, we'll run object detection on a live camera stream.


##
<p align="right">Next | <b><a href="detectnet-camera.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="detectnet-snapshot.md">Downloading the Detection Model to Jetson</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
