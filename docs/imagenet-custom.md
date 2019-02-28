<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-snapshot.md">Back</a> | <a href="detectnet-training.md">Next</a> | </sup><b><a href="../README.md"><sup>Contents</sup></a></b>
<br/>
<sup>Image Recognition</sup></p> 

# Loading Custom Models on Jetson

The `imagenet-console` and `imagenet-camera` programs that we used before also accept extended command line parameters for loading a custom model snapshot.  Set the `$NET` variable below to the path to your extracted snapshot:

``` bash
$ NET=networks/GoogleNet-ILSVRC12-subset

$ ./imagenet-console bird_0.jpg output_0.jpg \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_184080.caffemodel \
--labels=$NET/labels.txt \
--input_blob=data \
--output_blob=softmax
```

As before, the classification and confidence will be overlayed to the output image.  When compared to the output of the original network, the re-trained GoogleNet-12 makes similar classifications to the original GoogleNet-1000:

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-tensorRT-console-bird.png)

The extended command line parameters above also load custom classification models with [`imagenet-camera`](imagenet-camera/imagenet-camera.cpp). 

##
<p align="right">Next | <b><a href="detectnet-training.md">Locating Object Coordinates using DetectNet</a></b>
<br/>
Back | <b><a href="imagenet-snapshot.md">Downloading Model Snapshots to Jetson</a></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><b><a href="../README.md"><sup>Table of Contents</sup></a></b></p>