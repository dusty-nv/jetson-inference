<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-camera-2.md">Back</a> | <a href="detectnet-camera-2.md">Next</a> | </sup><a href="../README.md"><sup>Contents</sup></a>
<br/>
<sup>Object Detection</sup></s></p>

# Locating Object Coordinates using DetectNet
The previous image recognition examples output class probabilities representing the entire input image.   The second deep learning capability we're highlighting in this tutorial is detecting objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes).  This is performed using a 'detectNet' - or object detection / localization network.

The [`detectNet`](detectNet.h) object accepts as input the 2D image, and outputs a list of coordinates of the detected bounding boxes.  To train the object detection model, first a pretrained ImageNet recognition model (like Googlenet) is used with bounding coordinate labels included in the training dataset in addition to the source imagery.

The following pretrained DetectNet models are included with the tutorial:

1. **ped-100**        (single-class pedestrian detector)
2. **multiped-500**   (multi-class pedestrian + baggage detector)
3. **facenet-120**    (single-class facial recognition detector)
4. **coco-airplane**  (MS COCO airplane class)
5. **coco-bottle**    (MS COCO bottle class)
6. **coco-chair**     (MS COCO chair class)
7. **coco-dog**       (MS COCO dog class)

As with the previous examples, provided are a console program and a camera streaming program for using detectNet.

### Detecting Objects from the Command Line

The [`detectnet-console`](detectnet-console/detectnet-console.cpp) program can be used to find objects in images.  To load one of the pretrained object detection models that comes with the repo, you can specify the pretrained model name as the 3rd argument to `detectnet-console`:

``` bash
$ ./detectnet-console dog_1.jpg output_1.jpg coco-dog
```

The above command will process dog_1.jpg, saving it to output_1.jpg, using the pretrained DetectNet-COCO-Dog model.  This is a shortcut of sorts so you don't need to train the model yourself if you don't want to.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-1.jpg)

#### Pretrained DetectNet Models Available

Below is a table of the pretrained DetectNet snapshots downloaded with the repo (located in the `data/networks` directory after running `cmake` step) and the associated argument to `detectnet-console` used for loading the pretrained model:

| DIGITS model            | CLI argument    | classes              |
| ------------------------|-----------------|----------------------|
| DetectNet-COCO-Airplane | `coco-airplane` | airplanes            |
| DetectNet-COCO-Bottle   | `coco-bottle`   | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`    | chairs               |
| DetectNet-COCO-Dog      | `coco-dog`      | dogs                 |
| ped-100                 | `pednet`        | pedestrians          |
| multiped-500            | `multiped`      | pedestrians, luggage |
| facenet-120             | `facenet`       | faces                |

These all also have the python layer patch above already applied.

#### Running Other MS-COCO Models on Jetson

Let's try running some of the other COCO models.  The training data for these are all included in the dataset downloaded above.  Although the DIGITS training example above was for the coco-dog model, the same procedure can be followed to train DetectNet on the other classes included in the sample COCO dataset.

``` bash
$ ./detectnet-console bottle_0.jpg output_2.jpg coco-bottle
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-bottle-0.jpg)


``` bash
$ ./detectnet-console airplane_0.jpg output_3.jpg coco-airplane
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-airplane-0.jpg)

#### Running Pedestrian Models on Jetson

Included in the repo are also DetectNet models pretrained to detect humans.  The `pednet` and `multiped` models recognized pedestrians while `facenet` recognizes faces (from [FDDB](http://vis-www.cs.umass.edu/fddb/)).  Here's an example of detecting multiple humans simultaneously in a crowded space:


``` bash
$ ./detectnet-console peds-004.jpg output-4.jpg multiped
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-00.jpg" width="900">

### Multi-class Object Detection Models
When using the multiped model (`PEDNET_MULTI`), for images containing luggage or baggage in addition to pedestrians, the 2nd object class is rendered with a green overlay.

``` bash
$ ./detectnet-console peds-003.jpg output-3.jpg multiped
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-01.jpg" width="900">

Next, we'll run object detection on a live camera stream.

##
<p align="right">Next | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><b><a href="../README.md"><sup>Table of Contents</sup></a></b></p>