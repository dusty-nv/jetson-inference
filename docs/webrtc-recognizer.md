<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="webrtc-dash.md">Back</a> | <a href="aux-streaming.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>WebApp Frameworks</sup></s></p>

# Recognizer (Interactive Training)

The Recognizer is a Flask-based video tagging/classification webapp with interactive data collection and training.  As video is tagged and recorded, an updated model is incrementally re-trained in the background with PyTorch and then used for inference with TensorRT.  Both inference and training can run simultaneously, and the re-trained models are dynamically loaded at runtime for inference.

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-recognizer.jpg" width="600">

It also supports multi-label tagging, and in addition to recording client video over WebRTC, existing images can be uploaded from the client. The main source files for this example (found under [`python/www/recognizer`](../python/www/recognizer)) are as follows:

  * [`app.py`](../python/www/recognizer/app.py) (webserver)
  * [`stream.py`](../python/www/recognizer/stream.py) (WebRTC streaming thread)
  * [`model.py`](../python/www/recognizer/model.py) (DNN inferencing + training)
  * [`dataset.py`](../python/www/recognizer/dataset.py) (data tagging + recording)
  * [`index.html`](../python/www/recognizer/templates/index.html) (frontend presentation)

## Running the Example

Launching app.py will start a Flask webserver, a streaming thread that runs WebRTC and inferencing, and a training thread for PyTorch:

``` bash
$ cd jetson-inference/python/www/recognizer
$ pip3 install -r requirements.txt
$ python3 app.py --data=data/my_dataset
```

> **note**: receiving browser webcams requires [HTTPS/SSL](webrtc-server.md#enabling-https--ssl) to be enabled

The `--data` argument sets the path where your dataset and models are stored under.  If you built jetson-inference from source, you should elect to [Install PyTorch](building-repo-2.md#installing-pytorch) (or run the `install-pytorch.sh` script again). If you're using the Docker container, PyTorch is already installed for you.

After running app.py, you should be able to navigate your browser to `https://<JETSON-IP>:8050` and start the stream.  The default port is 8050, but you can change that with the `--port=N` command-line argument.  It's configured by default for WebRTC input and output, but if you want to use a different [video input device](aux-streaming.md#input-streams), you can set that with the `--input` argument (for example, `--input=/dev/video0` for a V4L2 camera)

### Collecting Data

If needed first select a client camera from the stream source dropdown on the webpage, and press the `Send` button.  When ready, enter class tag(s) of what the camera is looking at in the Tags selection box.  Once a tag is entered, you'll be able to either Record or Upload images into the dataset.  You can hold down the Record button to capture a video sequence.  Below is a high-level diagram of the data flow:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/webrtc-recognizer-diagram.jpg">

It's recommended to keep the distribution of tags across the classes relatively balanced - otherwise the model will be more likely to be biased towards certain classes.  You can view the label distribution and number of images in the dataset by expanding the `Training` dropdown.

### Training

As you add and tag new data, training can be enabled under the `Training` dropdown.  The training progress and accuracy will be updated on the page.  At the end of each epoch, if the model has the highest accuracy it will be exported to ONNX and loaded into TensorRT for inference.

There are various command-line options for the training that you can set when starting app.py:

| CLI Argument        | Description                                                                                                 | Default    |
|---------------------|-------------------------------------------------------------------------------------------------------------|------------|
| `--data`            | Path to where the data and models will be stored                                                            | `data/`    |
| `--net`             | The DNN architecture (see [here](https://pytorch.org/vision/stable/models.html#classification) for options) | `resnet18` |
| `--net-width`       | The width of the model (increase for higher accuracy)                                                       | 224        |
| `--net-height`      | The height of the model (increase for higher accuracy)                                                      | 224        |
| `--batch-size`      | Training batch size                                                                                         | 1          |
| `--workers`         | Number of dataloader threads                                                                                | 1          |
| `--optimizer`       | The solver (`adam` or `sgd`)                                                                                | `adam`     |
| `--learning-rate`   | Initial optimizer learning rate                                                                             | 0.001      |
| `--no-augmentation` | Disable color jitter / random flips on the training data                                                    | Enabled    |
  

### Inference

Inference can be enabled under the `Classification` dropdown.  When multi-label classification is used (i.e. the dataset contains images with multiple tags), all classification results will be shown that have confidence scores above the threshold that can be controlled from the page.

The app can be extended to trigger actions when certain objects are detected by adding your own code to the [`Model.Classify()`](https://github.com/dusty-nv/jetson-inference/blob/3476b4896051929f764f6b806378271dc82f23f1/python/www/recognizer/model.py#L83) function:

``` bash
def Classify(self, img):
   """
   Run classification inference and return the results.
   """
   if not self.inference_enabled:
      return
	  
   # returns a list of (classID, confidence) tuples
   self.results = self.model_infer.Classify(img, topK=0 if self.dataset.multi_label else 1)

   # to trigger custom actions/processing, add them here:
   for classID, confidence in self.results:
      if self.model_infer.GetClassLabel(classID) == 'person':             # update for your classes
         print(f"detected a person with {confidence * 100}% confidence")  # do something in response
   
   return self.results
```

When modifying backend server-side Python code, remember to restart app.py for changes to take effect.  As with the previous Flask example, various [REST queries](https://github.com/dusty-nv/jetson-inference/blob/master/docs/webrtc-flask.md#rest-queries) are used for communicating dynamic settings and state changes between the client and server, which you can also add to.

<p align="right">Next | <b><a href="aux-streaming.md">Camera Streaming and Multimedia</a></b>
<br/>
Back | <b><a href="webrtc-dash.md">Plotly Dashboard</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
