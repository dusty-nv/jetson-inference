<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">
<p align="right"><sup><a href="detectnet-camera-2.md">Back</a> | <a href="pytorch-cat-dog.md">Next</a> | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning</sup></s></p>

# Transfer Learning with PyTorch

Transfer learning is a technique for re-training a DNN model on a new dataset, which takes less time than training a network from scratch.  With transfer learning, the weights of a pre-trained model are fine-tuned to classify a customized dataset.  Although training is typically performed on a PC, server, or cloud instance with access to discrete GPU(s) due to the often large datasets and associated computational demands, by using transfer learning we're able to re-train various networks onboard Jetson to get started with training our own models.  <a href=https://pytorch.org/>PyTorch</a> is the machine learning framework that we'll be using, and example datasets and training scripts are provided to use below, along with a tool for collecting your own data.  

## Installing PyTorch

...

## Training Datasets

...

<p align="right">Next | <b><a href="pytorch-cat-dog.md">Training the Cat/Dog Dataset</a></b>
<br/>
Back | <b><a href="detectnet-camera-2.md">Running the Live Camera Detection Demo</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#hello-ai-world-inference-only"><sup>Table of Contents</sup></a></p>
