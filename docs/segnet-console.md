<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-patches.md">Back</a> | <a href="../README.md#two-days-to-a-demo-digits">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# Running Segmentation Models on Jetson

To test a custom segmentation network model snapshot on the Jetson, use the command line interface to [`segnet-console`](../examples/segnet-console/segnet-console.cpp)  

First, for convienience, set the path to your extracted snapshot into a `$NET` variable:

``` bash
$ NET=20170421-122956-f7c0_epoch_5.0

$ ./segnet-console drone_0428.png output_0428.png \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_22610.caffemodel \
--labels=$NET/fpv-labels.txt \
--colors=$NET/fpv-deploy-colors.txt \
--input_blob=data \ 
--output_blob=score_fr
```

This runs the specified segmentation model on a test image downloaded with the repo.

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-aerial-tensorRT.png)

In addition to the aerial model from this tutorial, the repo also includes pre-trained models on other segmentation datasets, including **[Cityscapes](https://www.cityscapes-dataset.com/)**, **[SYNTHIA](http://synthia-dataset.net/)**, and **[Pascal-VOC](http://host.robots.ox.ac.uk/pascal/VOC/)**.

##
<p align="right">Back | <b><a href="segnet-training.md">FCN-Alexnet Patches for TensorRT</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
