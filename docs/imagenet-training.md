<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-camera.md">Back</a> | <a href="imagenet-snapshot.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Image Recognition</sup></p> 

# Re-Training the Recognition Network

The existing GoogleNet and AlexNet models that are downloaded by the repo are pre-trained on [1000 classes of objects](../data/networks/ilsvrc12_synset_words.txt) from the ImageNet ILSVRC12 benchmark.

To recognize a new object class, you can use DIGITS to re-train the network on new data.  You can also organize the existing classes differently, including group multiple subclasses into one.  For example in this tutorial we'll take 230 of the 1000 classes, group those into 12 classes and retrain the network.

Let's start by downloading the ILSVRC12 images to work with, or you can substitute your own dataset in an **[Image Folder](https://github.com/NVIDIA/DIGITS/blob/master/docs/ImageFolderFormat.md)**.

### Downloading Image Recognition Dataset

An image recognition dataset consists of a large number of images sorted by their classification type (typically by directory).  The ILSVRC12 dataset was used in the training of the default GoogleNet and AlexNet models.  It's roughly 100GB in size and includes 1 million images over 1000 different classes.  The dataset is downloaded to the DIGITS server using the [`imagenet-download.py`](../tools/imagenet-download.py) image crawler.

To download the dataset, first make sure you have enough disk space on your DIGITS server (120GB recommended), then run the following commands from a directory on that machine where you want the dataset stored:

``` bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/gzr5iewf5aouhc5exhp3higw6lzhcysj.gz -O ilsvrc12_urls.tar.gz
$ tar -xzvf ilsvrc12_urls.tar.gz
$ wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-download.py
$ python imagenet-download.py ilsvrc12_urls.txt . --jobs 100 --retry 3 --sleep 0
```

In the commands above the list of image URLs along with the scripts are downloaded before launching the crawler.

> **note**: be considerate running the image crawler from a corporate network, IT may flag the activity.
> It will probably take overnight on a decent connection to download the 1000 ILSVRC12 classes (100GB).

The crawler will download images to subdirectories that correspond to it's classification.  Each image class is stored in it's own directory, with 1000 directories in total (one for each class in ILSVRC12).  The folders are organized with a naming scheme similar to:

```
n01440764/
n01443537/
n01484850/
n01491361/
n01494475/
...
```

These 8-digit ID's prefixed wth N are referred to as the **synset ID** of the class.  The name string of the class can be looked up in [`ilsvrc12_synset_words.txt`](../data/networks/ilsvrc12_synset_words.txt).  For example, synset `n01484850 great white shark`.

### Customizing the Object Classes

The dataset that we downloaded in the previous step was used to train the default AlexNet and GoogleNet models with 1000 object classes from several core groups, including different species of birds, plants, fruit, and fish, dog and cat breeds, types of vehicles, ect.  For practicle purposes lets consider a companion to the GoogleNet model which recognizes a dozen core groups made up of the original 1000 classes (for example, instead of detecting 122 individual breeds of dogs, combining them all into one common `dog` class).  These 12 core groups may be more practical to use than 1000 individual synsets and combining across classes results in more training data and stronger classification for the group.

DIGITS expects the data in a hierarchy of folders, so we can create directories for the groups and then symbolically link to the synsets from ILSVRC12 downloaded above.  DIGITS will automatically combine images from all folders under the top-level groups.  The directory structure resembles the following, with the value in parenthesis indicates the number of classes used to make up the group and the value next to the arrows indicating the synset ID linked to.

```
‣ ball/  (7)
	• baseball     (→n02799071)
	• basketball   (→n02802426)
	• soccer ball  (→n04254680)
	• tennis ball  (→n04409515)
	• ...
‣ bear/  (4)
	• brown bear   (→n02132136)
	• black bear   (→n02133161)
	• polar bear   (→n02134084)
	• sloth bear   (→n02134418)
• bike/  (3)
• bird/  (17)
• bottle/ (7)
• cat/  (13)
• dog/  (122)
• fish/   (5)
• fruit/  (12)
• turtle/  (5)
• vehicle/ (14)
• sign/  (2)
```

Since there are actually a lot of synsets linked to from ILSVRC12, we provide the **[`imagenet-subset.sh`](../tools/imagenet-subset.sh)** script to generate the directory structure and links given the path to the dataset.  Run the folowing commands from the DIGITS server:

``` bash
$ wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-subset.sh
$ chmod +x imagenet-subset.sh
$ mkdir 12_classes
$ ./imagenet-subset.sh /opt/datasets/imagenet/ilsvrc12 12_classes
```

In this example the links are created in the `12_classes` folder, with the first argument to the script being the path to ILSVRC12 downloaded in the previous step. 

### Importing Classification Dataset into DIGITS

Navigate your browser to your DIGITS server instance and choose to create a new `Classification Dataset` from the drop-down under the Datasets tab:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-dataset-menu.png" width="250">

Set the `Training Images` path to the `12_classes` folder from the previous step and make the following

* % for validation:  `10`
* Group Name:  `ImageNet`
* Dataset Name: `ImageNet-ILSVRC12-subset`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-dataset.png)

Use the `Create` button at the bottom of the page to launch the dataset import job.  The size of the data subset is around 20GB, so depending on server I/O performance it takes 10-15 minutes.  Next we'll create the new model and begin training it.

### Creating Image Classification Model with DIGITS

When the previous data import job is complete, return to the DIGITS home screen.  Select the `Models` tab and choose to create a new `Classification Model` from the drop-down:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-model-menu.png" width="250">

Make the following settings in the form:

* Select Dataset:  `ImageNet-ILSVRC12-subset`
* Subtract Mean:  `Pixel`
* Standard Networks:  `GoogleNet`
* Group Name:  `ImageNet`
* Model Name:  `GoogleNet-ILSVRC12-subset`

After selecting a GPU to train on, click the `Create` button at the bottom to begin training.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-model.png)

### Testing Classification Model in DIGITS

After the training job completes 30 epochs, the trained model should appear like so:

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-model.png)

At this point, we can try testing our new model's inference on some example images in DIGITS.  On the same page as the plot above, scroll down under the `Trained Models` section.  Under `Test a Single Image`, select an image to try (for example, `/ilsvrc12/n02127052/n02127052_1203.jpg`):

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-test-single-image.png" width="350">

Press the `Classify One` button and you should see a page similar to:

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-infer-cat.png)

The image is classified as the new GoogleNet-12 model as `cat`, while in the original GoogleNet-1000 it was under `Lynx`.  This indicates the new model is working ok, because the Lynx category was included in GoogleNet-12's training of cat.

##
<p align="right">Next | <b><a href="imagenet-snapshot.md">Downloading Model Snapshots to Jetson</a></b>
<br/>
Back | <b><a href="imagenet-camera.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
