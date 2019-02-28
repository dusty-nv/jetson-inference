<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="imagenet-training.md">Back</a> | <a href="imagenet-custom.md">Next</a> | </sup><b><a href="../README.md"><sup>Contents</sup></a></b>
<br/>
<sup>Image Recognition</sup></p> 

# Downloading Model Snapshots to Jetson

Now that we confirmed the trained model is working in DIGITS, let's download and extract the model snapshot to Jetson.  From the browser on your Jetson TX1/TX2, navigate to your DIGITS server and the `GoogleNet-ILSVRC12-subset` model.  Under the `Trained Models` section, select the desired snapshot from the drop-down (usually the one with the highest epoch) and click the `Download Model` button.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-model-download.png" width="650">

Alternatively, if your Jetson and DIGITS server aren't accessible from the same network, you can use the step above to download the snapshot to an intermediary machine and then use SCP or USB stick to copy it to Jetson.  

Then extract the archive with a command similar to:

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170524-140310-8c0b_epoch_30.0.tar.gz
```

Next we will load our custom snapshot into TensorRT, running on the Jetson.

##
<p align="right">Next | <b><a href="imagenet-custom.md">Loading Custom Models on Jetson</a></b>
<br/>
Back | <b><a href="imagenet-training.md">Re-training the Recognition Network</a></p>
<p align="center"><sup>© 2016-2019 NVIDIA | </sup><b><a href="../README.md"><sup>Table of Contents</sup></a></b></p>