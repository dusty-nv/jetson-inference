<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# What's Deep Learning?

### Introduction

*Deep-learning* networks typically have two primary phases of development:   **training** and **inference**

#### Training
During the training phase, the network learns from a large dataset of labeled examples.  The weights of the neural network become optimized to recognize the patterns contained within the training dataset.  Deep neural networks have many layers of neurons connected togethers.  Deeper networks take increasingly longer to train and evaluate, but are ultimately able to encode more intelligence within them.

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/fd4ba9e7e68b76fc41c8312856c7d0ad)

Throughout training, the network's inference performance is tested and refined using trial dataset. Like the training dataset, the trial dataset is labeled with ground-truth so the network's accuracy can be evaluated, but was not included in the training dataset.  The network continues to train iteratively until it reaches a certain level of accuracy set by the user.

Due to the size of the datasets and deep inference networks, training is typically very resource-intensive and can take weeks or months on traditional compute architectures.  However, using GPUs vastly accellerates the process down to days or hours.  

##### DIGITS

Using [DIGITS](https://developer.nvidia.com/digits), anyone can easily get started and interactively train their networks with GPU acceleration.  <br />DIGITS is an open-source project contributed by NVIDIA, located here: https://github.com/NVIDIA/DIGITS. 

This tutorial will use DIGITS and Jetson TX1 together for training and deploying deep-learning networks, <br />refered to as the DIGITS workflow:

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/90bde1f85a952157b914f75a9f8739c2)


#### Inference
Using it's trained weights, the network evaluates live data at runtime.  Called inference, the network predicts and applies reasoning based off the examples it learned.  Due to the depth of deep learning networks, inference requires significant compute resources to process in realtime on imagery and other sensor data.  However, using NVIDIA's GPU Inference Engine which uses Jetson's integrated NVIDIA GPU, inference can be deployed onboard embedded platforms.  Applications in robotics like picking, autonomous navigation, agriculture, and industrial inspection have many uses for deploying deep inference, including:

  - Image recognition
  - Object detection
  - Segmentation 
  - Image registration (homography estimation)
  - Depth from raw stereo
  - Signal analytics
  