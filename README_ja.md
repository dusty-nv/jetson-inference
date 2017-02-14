![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/841b9209217f74e5992b8d332c612126)
# ディープラーニングをデプロイする（実世界で使えるようにする）
こちらはNVIDIAの推論と **[Jetson TX1](http://www.nvidia.com/object/embedded-systems.html)** 用の組込みディープビジョン・ランタイムライブラリを使うためのガイドです.

NVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)** を使ってニューラルネットワークを効率的に現場にデプロイするためのリソースがこのレポジトリに含まれています。

ビジョン用のプリミティブ、例えば画像認識用の [`imageNet`](imageNet.h)や、物体検出用の [`detectNet`](detectNet.h)、そしてセグメンテーション用の [`segNet`](segNet.h) は、共通の [`tensorNet`](tensorNet.h) オブジェクトを継承しています。サンプルとしては、ライブカメラからのストリームしたりディスクからの画像を処理する例が提供されています。これらを理解して応用するための道筋を10個のステップにまとめました。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-primitives.png" width="800">

### **ディープラーニングまでの10ステップ**

1. [ディープラーニングとは？](#whats-deep-learning)
2. [JetPack 2.3 / TensorRT を入手](#getting-tensorrt)
3. [ソースからビルド](#building-from-source)
4. [コード詳説](#digging-into-the-code)
5. [Classify Images with ImageNet](#classifying-images-with-imagenet)
6. [Run the Live Camera Recognition Demo](#running-the-live-camera-recognition-demo)
7. [Re-train the Network with Customized Data](#re-training-the-network-with-customized-data)
8. [Locate Object Coordinates using DetectNet](#locating-object-coordinates-using-detectNet)
9. [Run the Live Camera Detection Demo](#running-the-live-camera-detection-demo)
10. [Re-train DetectNet with DIGITS](#re-training-detectnet-with-digits)


**Recommended System Requirements**

学習用 GPU: Maxwell 世代もしくは Pascal 世代の TITAN-X、Tesla M40、Tesla P40、もしくは AWS P2 インスタンス。  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 14.04 x86_64 もしくは Ubuntu 16.04 x86_64 ( DIGITS [AWS AMI](https://aws.amazon.com/marketplace/pp/B01LZN28VD) イメージを参照).

エッジ側:    &nbsp;&nbsp;Jetson TX1 開発キット、JetPack 2.3 かそれ以降 (Ubuntu 16.04 aarch64).

> **注意**:  この [ブランチ](http://github.com/dusty-nv/jetson-inference) は Jetson TX1 と以下のBSPの組み合わせで検証されています: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)

>  別のブランチもあるので注意: [JetPack 2.2 / L4T R24.1 aarch64 (Ubuntu 14.04 LTS)](http://github.com/dusty-nv/jetson-inference/tree/L4T-R24.1) 

レポジトリに含まれる TensorRT のサンプルは組込みの Jetson TX1 モジュール用ですが、cuDNN と TensorRT がホスト側にインストールされている場合は TensorRT をホスト側のPCでコンパイルすることも可能です。

## ディープラーニングとは？

ディープ・ニューラルネットワーク（DNN）や機械学習といった言葉が初めてという方は、こちらの [入門テキスト](docs/deep-learning.md) をご覧ください。

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/7aca8779d265a860d5133cdc8c6c6b76" width="800"></a>

NVIDIAのディープラーニング用のツールを使えば、簡単にDNNの学習を **[始めたり](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)** 高性能にデプロイすることが可能です。

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/5720072a6941032685ea18c4e4068a23" width="700"></a>

NVIDIA [DIGITS](https://github.com/NVIDIA/DIGITS) はクラウドやPC上のラベル付けされたデータセットに対してネットワークモデルの学習をインタラクティブに行えるツールです。一方 TensorRT や Jetson は推論ランタイムを現場にデプロイするのに用います。DIGITs と TensorRT を一緒に使うことで、高度なAIや認識を実現しうるディープ・ニューラルネットワークを開発・デプロイするための非常に効率のいいワークフローを実現できます。 

DIGITS [スタートガイド](https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md) を読んだあとに、このチュートリアルの次の章 [ を入手](#getting-tensorrt) を読んでください。

最新の DIGITS をNVIDIAのGPUを搭載したホストPC もしくはクラウドサービスにインストールしてください。こちら [developer.nvidia.com/digits](http://developer.nvidia.com/digits) からビルド済みのDockerイメージ、もしくは Amazon Machine Image (AMI) を参照ください。

## TensorRT を入手

NVIDIA TensorRT は **[JetPack 2.3](https://developer.nvidia.com/embedded/jetpack)** から利用可能になった新しいライブラリで、プロダクトレベルのDNNの最適化とデプロイのためのものです。TensorRT は多くのグラフ最適化が施され、また Tegra X1 で利用可能になった 半精度 FP16 を利用して、既存 Caffe 二倍の性能を実現します。

<a href="https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/91d88749a582e884926686f7a9a7f9fd" width="700"></a>

ワットあたりの処理性能 (iages/sec/Watts) を測るベンチマークでは、TensorRT を実装した Jetson TX1 は、従来のCPUに比べて20倍も電力効率がいいことが示されました。技術概要についてはこちらの **[Parallel ForAll](https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/)**  ブログ記事をご覧ください。

<a href="https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-tx1-deep-learning-inference/"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/86d79898dbb3c0664ab1fcf112da4e6e" width="700"></a>

TensorRT を入手するには、最新の [JetPack](https://developer.nvidia.com/embedded/jetpack) をお手持ちのPCにダウンロードし、Jetson をフラッシュし直してください。(手順は [Jetson TX1 ユーザーガイド](http://developer.nvidia.com/embedded/dlc/l4t-24-1-jetson-tx1-user-guide)を参照).

## ソースからビルド
このレポジトリで提供されるのは、TensorRT を使ったサンプルプログラムで、Googlenet/Alexnet をカメラからの生映像に対してかけて画像認識を行ったり、歩行者検出を行いバウンディングボックスを描くものがあります。 

最新のソースコードは、[GitHub](http://github.com/dusty-nv/jetson-inference) から入手でき、Jetson TX1 上でコンパイルします。

> **注意**:  この [ブランチ](http://github.com/dusty-nv/jetson-inference) は以下の組み合わせで検証しています。 
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS)
      
#### 1. レポジトリをクローン
To obtain the repository, navigate to a folder of your choosing on the Jetson.  First, make sure git and cmake are installed locally:

``` bash
sudo apt-get install git cmake
```

Then clone the jetson-inference repo:
``` bash
git clone http://github.org/dusty-nv/jetson-inference
```

#### 2. 設定
When cmake is run, a special pre-installation script (CMakePreBuild.sh) is run and will automatically install any dependencies.

``` bash
cd jetson-inference
mkdir build
cd build
cmake ../
```

#### 3. コンパイル
Make sure you are still in the jetson-inference/build directory, created above in step #2.

``` bash
cd jetson-inference/build			# omit if pwd is already /build from above
make
```

Depending on architecture, the package will be built to either armhf or aarch64, with the following directory structure:

```
|-build
   \aarch64		    (64-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```

binaries residing in aarch64/bin, headers in aarch64/include, and libraries in aarch64/lib.

## コード詳説

For reference, see the available vision primitives, including [`imageNet`](imageNet.h) for image recognition and [`detectNet`](detectNet.h) for object localization.

``` c++
/**
 * Image recognition with GoogleNet/Alexnet or custom models, using TensorRT.
 */
class imageNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		ALEXNET,
		GOOGLENET
	};

	/**
	 * Load a new network instance
	 */
	static imageNet* Create( NetworkType networkType=GOOGLENET );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 */
	static imageNet* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
							 const char* class_labels, const char* input="data", const char* output="prob" );

	/**
	 * Determine the maximum likelihood image class.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );
};
```

Both inherit from the shared [`tensorNet`](tensorNet.h) object which contains common TensorRT code.

## Classifying Images with ImageNet
There are multiple types of deep learning networks available, including recognition, detection/localization, and soon segmentation.  The first deep learning capability to highlight is **image recognition** using an 'imageNet' that's been trained to identify similar objects.

The [`imageNet`](imageNet.h) object accept an input image and outputs the probability for each class.  Having been trained on ImageNet database of **[1000 objects](data/networks/ilsvrc12_synset_words.txt)**, the standard AlexNet and GoogleNet networks are downloaded during [step 2](#configuring) from above.

After building, first make sure your terminal is located in the aarch64/bin directory:

``` bash
$ cd jetson-inference/build/aarch64/bin
```

Then, classify an example image with the [`imagenet-console`](imagenet-console/imagenet-console.cpp) program.  [`imagenet-console`](imagenet-console/imagenet-console.cpp) accepts 2 command-line arguments:  the path to the input image and path to the output image (with the class overlay printed).

``` bash
$ ./imagenet-console orange_0.jpg output_0.jpg
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/8c63ed0975b4c89a4134c320d4e47931"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/8c63ed0975b4c89a4134c320d4e47931" width="700"></a>

``` bash
$ ./imagenet-console granny_smith_1.jpg output_1.jpg
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/b6aea9d50490fbe261420ab940de0efd"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/b6aea9d50490fbe261420ab940de0efd" width="700"></a>

Next, we will use [imageNet](imageNet.h) to classify a live video feed from the Jetson onboard camera.

## Running the Live Camera Recognition Demo

Similar to the last example, the realtime image recognition demo is located in /aarch64/bin and is called [`imagenet-camera`](imagenet-camera/imagenet-camera.cpp).
It runs on live camera stream and depending on user arguments, loads googlenet or alexnet with TensorRT. 
``` bash
$ ./imagenet-camera googlenet           # to run using googlenet
$ ./imagenet-camera alexnet             # to run using alexnet
```

The frames per second (FPS), classified object name from the video, and confidence of the classified object are printed to the openGL window title bar.  By default the application can recognize up to 1000 different types of objects, since Googlenet and Alexnet are trained on the ILSVRC12 ImageNet database which contains 1000 classes of objects.  The mapping of names for the 1000 types of objects, you can find included in the repo under [data/networks/ilsvrc12_synset_words.txt](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)

> **note**:  by default, the Jetson's onboard CSI camera will be used as the video source.  If you wish to use a USB webcam instead, change the `DEFAULT_CAMERA` define at the top of [`imagenet-camera.cpp`](imagenet-camera/imagenet-camera.cpp) to reflect the /dev/video V4L2 device of your USB camera.  The model it's tested with is Logitech C920. 

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/399176be3f3ab2d9bfade84e0afe2abd" width="800"></a>
<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/93071639e44913b6f23c23db2a077da3" width="800"></a>

## Re-training the Network with Customized Data

The existing GoogleNet and AlexNet models that are downloaded by the repo are pre-trained on [1000 classes of objects](data/networks/ilsvrc12_synset_words.txt).

What if you require a new object class to be added to the network, or otherwise require a different organization of the classes?  

Using [NVIDIA DIGITS](http://github.com/NVIDIA/DIGITS), networks can be fine-tuned or re-trained from a pre-exisiting network model.
After installing DIGITS on a PC or in the cloud (such as an AWS instance), see the **[Image Folder Specification](https://github.com/NVIDIA/DIGITS/blob/master/docs/ImageFolderFormat.md)** to learn how to organize the data for your particular application.

Popular training databases with various annotations and labels include [ImageNet](image-net.org), [MS COCO](mscoco.org), and [Google Images](images.google.com) among others.

See [here](http://www.deepdetect.com/tutorials/train-imagenet/) under the `Downloading the dataset` section to obtain a crawler script that will download the 1000 original classes, including as many of the original images that are still available online.

> **note**: be considerate running the crawler script from a corporate network, they may flag the activity.
> It will probably take overnight on a decent connection to download the 1000 ILSVRC12 classes (100GB) from ImageNet (1.2TB)

Then, while creating the new network model in DIGITS, copy the [GoogleNet prototxt](data/networks/googlenet.prototxt) and specify the existing GoogleNet caffemodel as the DIGITS **Pretrained Model**:

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/610745a8bafae4a5686d45901f5cc6f3"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/610745a8bafae4a5686d45901f5cc6f3" width="800"></a>

The network training should now converge faster than if it were trained from scratch.  After the desired accuracy has been reached, copy the new model checkpoint back over to your Jetson and proceed as before, but now with the added classes available for recognition.

## Locating Object Coordinates using DetectNet
The previous image recognition examples output class probabilities representing the entire input image.   The second deep learning capability to highlight is detecting multiple objects, and finding where in the video those objects are located (i.e. extracting their bounding boxes).  This is performed using a 'detectNet' - or object detection / localization network.

The [`detectNet`](detectNet.h) object accepts as input the 2D image, and outputs a list of coordinates of the detected bounding boxes.  Three example detection network models are are automatically downloaded during the repo [source configuration](#configuring):

1. **ped-100**  (single-class pedestrian detector)
2. **multiped-500**   (multi-class pedestrian + baggage detector)
3. **facenet-120**  (single-class facial recognition detector)

To process test images with [`detectNet`](detectNet.h) and TensorRT, use the [`detectnet-console`](detectnet-console/detectnet-console.cpp) program.  [`detectnet-console`](detectnet-console/detectnet-console.cpp) accepts command-line arguments representing the path to the input image and path to the output image (with the bounding box overlays rendered).  Some test images are included with the repo:

``` bash
$ ./detectnet-console peds-007.png output-7.png
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/eb1066d317406abb66be939e23150ccc"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/eb1066d317406abb66be939e23150ccc" width="900"></a>

To change the network that [`detectnet-console`](detectnet-console/detectnet-console.cpp) uses, modify [`detectnet-console.cpp`](detectnet-console/detectnet-console.cpp) (beginning line 33):
``` c
detectNet* net = detectNet::Create( detectNet::PEDNET_MULTI );	 // uncomment to enable one of these 
//detectNet* net = detectNet::Create( detectNet::PEDNET );
//detectNet* net = detectNet::Create( detectNet::FACENET );
```
Then to recompile, navigate to the `jetson-inference/build` directory and run `make`.
### Multi-class Object Detection
When using the multiped-500 model (`PEDNET_MULTI`), for images containing luggage or baggage in addition to pedestrians, the 2nd object class is rendered with a green overlay.
``` bash
$ ./detectnet-console peds-008.png output-8.png
```

<a href="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/c0c41b17fb6ea05315b64f3ee7cbbb84"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/c0c41b17fb6ea05315b64f3ee7cbbb84" width="900"></a>

## Running the Live Camera Detection Demo

Similar to the previous example, [`detectnet-camera`](detectnet-camera/detectnet-camera.cpp) runs the object detection networks on live video feed from the Jetson onboard camera.  Launch it from command line along with the type of desired network:

``` bash
$ ./detectnet-camera multiped       # run using multi-class pedestrian/luggage detector
$ ./detectnet-camera ped-100        # run using original single-class pedestrian detector
$ ./detectnet-camera facenet        # run using facial recognition network
$ ./detectnet-camera                # by default, program will run using multiped
```

> **note**:  to achieve maximum performance while running detectnet, increase the Jetson TX1 clock limits by running the script:
>  `sudo ~/jetson_clocks.sh`
<br/>
> **note**:  by default, the Jetson's onboard CSI camera will be used as the video source.  If you wish to use a USB webcam instead, change the `DEFAULT_CAMERA` define at the top of [`detectnet-camera.cpp`](detectnet-camera/detectnet-camera.cpp) to reflect the /dev/video V4L2 device of your USB camera.  The model it's tested with is Logitech C920.  

## Re-training DetectNet with DIGITS

For a step-by-step guide to training custom DetectNets, see the **[Object Detection](https://github.com/NVIDIA/DIGITS/tree/digits-4.0/examples/object-detection)** example included in DIGITS version 4:

<a href="https://github.com/NVIDIA/DIGITS/tree/digits-4.0/examples/object-detection"><img src="https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/0c1a5ee3ab9c4629ac61cbbe9aae3e10" width="500"></a>

The DIGITS guide above uses the [KITTI](http://www.cvlibs.net/datasets/kitti/) dataset, however [MS COCO](http://mscoco.org) also has bounding data available for a variety of objects.

## Extra Resources

In this area, links and resources for deep learning developers are listed:

* [Appendix](docs/aux-contents.md)
	* [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-institute) — [Introductory QwikLabs](https://developer.nvidia.com/deep-learning-courses)
     * [Building nvcaffe](docs/building-nvcaffe.md)
	* [Other Examples](docs/other-examples.md)
	* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes

