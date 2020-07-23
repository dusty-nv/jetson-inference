<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="segnet-training.md">Back</a> | <a href="segnet-console.md">Next</a> | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Contents</sup></a>
<br/>
<sup>Semantic Segmentation</sup></s></p>

# FCN-Alexnet Patches for TensorRT

There exist a couple non-essential layers included in the original FCN-Alexnet which aren't supported in TensorRT and should be deleted from the `deploy.prototxt` included in the snapshot. 

At the end of `deploy.prototxt`, delete the deconv and crop layers:

```
layer {
  name: "upscore"
  type: "Deconvolution"
  bottom: "score_fr"
  top: "upscore"
  param {
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 21
    bias_term: false
    kernel_size: 63
    group: 21
    stride: 32
    weight_filler {
      type: "bilinear"
    }
  }
}
layer {
  name: "score"
  type: "Crop"
  bottom: "upscore"
  bottom: "data"
  top: "score"
  crop_param {
    axis: 2
    offset: 18
  }
}
```

And on line 24 of `deploy.prototxt`, change `pad: 100` to `pad: 0`.  

Finally, copy the `fpv-labels.txt` and `fpv-deploy-colors.txt` from the aerial dataset to your model snapshot folder on Jetson.  Your FCN-Alexnet model snapshot is now compatible with TensorRT.  Now we can run it on Jetson and perform inference on images.

##
<p align="right">Next | <b><a href="segnet-console.md">Running Segmentation Models on Jetson</a></b>
<br/>
Back | <b><a href="segnet-training.md">Training FCN-Alexnet with DIGITS</a></p>
</b><p align="center"><sup>© 2016-2019 NVIDIA | </sup><a href="../README.md#two-days-to-a-demo-digits"><sup>Table of Contents</sup></a></p>
