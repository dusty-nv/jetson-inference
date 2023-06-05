<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="imagenet-camera-2.md">Back</a> | <a href="detectnet-console-2.md">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Image Classification</sup></p>

# Multi-Label Classification for Image Tagging

Multi-label classification models are able to recognize multiple object classes simultaneously for performing tasks like image tagging.  The multi-label DNNs are almost identical in topology to ordinary single-class models, except they use a sigmoid activation layer as opposed to softmax.  There's a pre-trained `resnet18-tagging-voc` multi-label model available that was trained on the Pascal VOC dataset:  

<img src=https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet_tagging.jpg>

To enable image tagging, you'll want to run imagenet/imagenet.py with `--topK=0` and a `--threshold` of your choosing: 

``` bash
# C++
$ imagenet --model=resnet18-tagging-voc --topK=0 --threshold=0.25 "images/object_*.jpg" images/test/tagging_%i.jpg

# Python
$ imagenet.py --model=resnet18-tagging-voc --topK=0 --threshold=0.25 "images/object_*.jpg" images/test/tagging_%i.jpg
```

Using `--topK=0` means that all the classes with a confidence score exceeding the threshold will be returned by the classifier.

### Retrieving Multiple Image Tags from Code

The imageNet.Classify() function will return `(classID, confidence)` tuples when the topK argument is specified.  See [`imagenet.cpp`](../examples/imagenet/imagenet.cpp) or [`imagenet.py`](../python/examples/imagenet.py) for code examples of using multiple classification results:

#### C++ 

``` cpp
imageNet::Classifications classifications;	// std::vector<std::pair<uint32_t, float>>  (classID, confidence)

if( net->Classify(image, input->GetWidth(), input->GetHeight(), classifications, topK) < 0 )
	continue;

for( uint32_t n=0; n < classifications.size(); n++ )
{
	const uint32_t classID = classifications[n].first;
	const char* classLabel = net->GetClassLabel(classID);
	const float confidence = classifications[n].second * 100.0f;
	
	printf("imagenet:  %2.5f%% class #%i (%s)\n", confidence, classID, classLabel);	
}
```

#### Python

``` python
predictions = net.Classify(img, topK=args.topK)

for n, (classID, confidence) in enumerate(predictions):
   classLabel = net.GetClassLabel(classID)
   confidence *= 100.0
   print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")
```

Note that topK can also be used in single-class classification to get the top N results, although those models weren't trained for image tagging.

<p align="right">Next | <b><a href="detectnet-console-2.md">Detecting Objects from Images</a></b>
<br/>
Back | <b><a href="imagenet-camera-2.md">Running the Live Camera Recognition Demo</a></p>
</b><p align="center"><sup>© 2016-2023 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>
