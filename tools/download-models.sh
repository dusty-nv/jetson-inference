#!/bin/bash
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

APP_TITLE="Hello AI World (jetson-inference)"
OUTPUT_DIR="../data/networks"
LOG="[jetson-inference] "
WGET_QUIET="--quiet"
BUILD_INTERACTIVE=${1:-"YES"}


#
# exit message for user
#
function exit_message()
{
	echo " "
	echo "$LOG to run this tool again, use the following commands:"
	echo " "
	echo "    $ cd <jetson-inference>/tools"
	echo "    $ ./download-models.sh"
	echo " "

	exit $1
}

#
# prompt user for retry
#
function retry_prompt()
{
	dialog --backtitle "$APP_TITLE" \
			  --title "Download Error" \
			  --colors \
			  --extra-button \
			  --extra-label "Next" \
			  --cancel-label "Quit" \
			  --ok-label "Retry" \
			  --yesno "\nFailed to download '$1' (error code=$2)\n\nWould you like to try downloading it again?\n\n\ZbNote:\Zn  if this error keeps occuring, see here:\n https://github.com/dusty-nv/jetson-inference/releases" 12 60

	local retry_status=$?
	clear

	WGET_QUIET="--verbose"

	if [ $retry_status = 1 ]; then
		echo "$LOG models failed to download (they may not load at runtime)"
		exit_message 1
	elif [ $retry_status != 0 ]; then
		return 1
	fi

	return 0
}


#
# try to download a file from URL
#
function attempt_download_file()
{
	local filename=$1
	local URL=$2
	
	wget $WGET_QUIET --show-progress --progress=bar:force:noscroll --no-check-certificate $URL -O $filename
	
	local wget_status=$?

	if [ $wget_status != 0 ]; then
		echo "$LOG wget failed to download '$filename' (error code=$wget_status)"
		return $wget_status
	fi

	mv $filename $OUTPUT_DIR
	return 0
}


#
# download a file from URL
#
function download_file()
{
	local filename=$1
	local URL=$2
	
	WGET_QUIET="--quiet"

	while true; do
		attempt_download_file $filename $URL

		local download_status=$?

		if [ $download_status = 0 ]; then
			return 0
		fi

		retry_prompt $filename $download_status
	
		local retry_status=$?

		if [ $retry_status != 0 ]; then
			return 0
		fi
	done
}


#
# try to download/extract an archive
#
function attempt_download_archive()
{
	local filename=$1
	local URL=$2
	
	wget $WGET_QUIET --show-progress --progress=bar:force:noscroll --no-check-certificate $URL -O $filename
	
	local wget_status=$?

	if [ $wget_status != 0 ]; then
		echo "$LOG wget failed to download '$filename' (error code=$wget_status)"
		return $wget_status
	fi

	tar -xzf $filename -C $OUTPUT_DIR

	local tar_status=$?

	if [ $tar_status != 0 ]; then
		echo "$LOG tar failed to extract '$filename' (error code=$tar_status)"
		return $tar_status
	fi

	rm $filename
	return 0
}


#
# download/extract an archive
#
function download_archive()
{
	local filename=$1
	local URL=$2
	
	WGET_QUIET="--quiet"

	while true; do
		attempt_download_archive $filename $URL

		local download_status=$?

		if [ $download_status = 0 ]; then
			return 0
		fi

		retry_prompt $filename $download_status
	
		local retry_status=$?

		if [ $retry_status != 0 ]; then
			return 0
		fi
	done
}

#
# IMAGE RECOGNITION
#
function download_alexnet()
{
	echo "$LOG Downloading AlexNet..."

	download_file "bvlc_alexnet.caffemodel" "https://nvidia.box.com/shared/static/5j264j7mky11q8emy4q14w3r8hl5v6zh.caffemodel"
	download_file "alexnet.prototxt" "https://nvidia.box.com/shared/static/c84wp3axbtv4e2gybn40jprdquav9azm.prototxt"
	download_file "alexnet_noprob.prototxt" "https://nvidia.box.com/shared/static/o0w0sl3obqxj21u09c0cwzw4khymz7hh.prototxt"
}

function download_googlenet()
{
	echo "$LOG Downloading GoogleNet..."

	download_file "bvlc_googlenet.caffemodel" "https://nvidia.box.com/shared/static/at8b1105ww1c5h7p30j5ko8qfnxrs0eg.caffemodel" 
	download_file "googlenet.prototxt" "https://nvidia.box.com/shared/static/5z3l76p8ap4n0o6rk7lyasdog9f14gc7.prototxt"
	download_file "googlenet_noprob.prototxt" "https://nvidia.box.com/shared/static/ue8qrqtglu36andbvobvaaj8egxjaoli.prototxt"
}

function download_googlenet12()
{
	echo "$LOG Downloading GoogleNet-12..."
	download_archive "GoogleNet-ILSVRC12-subset.tar.gz" "https://nvidia.box.com/shared/static/zb8i3zcg39sdjjxfty7o5935hpbd64y4.gz" 
}

function download_resnet18()
{
	echo "$LOG Downloading ResNet-18..."
	download_archive "ResNet-18.tar.gz" "https://nvidia.box.com/shared/static/gph1qfor89vh498op8cicvwc13zltu3h.gz" 
}

function download_resnet50()
{
	echo "$LOG Downloading ResNet-50..."
	download_archive "ResNet-50.tar.gz" "https://nvidia.box.com/shared/static/ht46fmnwvow0o0n0ke92x6bzkht8g5xb.gz" 
}

function download_resnet101()
{
	echo "$LOG Downloading ResNet-101..."
	download_archive "ResNet-101.tar.gz" "https://nvidia.box.com/shared/static/7zog25pu70nxjh2irni49e5ujlg4dl82.gz" 
}

function download_resnet152()
{
	echo "$LOG Downloading ResNet-152..."
	download_archive "ResNet-152.tar.gz" "https://nvidia.box.com/shared/static/6t621ru1i054vscvhx3rqck8597es7w8.gz" 
}

function download_vgg16()
{
	echo "$LOG Downloading VGG-16..."
	download_archive "VGG-16.tar.gz" "https://nvidia.box.com/shared/static/ar2ttdpnw1drzxnvpw0umzkw67fka3h0.gz" 
}

function download_vgg19()
{
	echo "$LOG Downloading VGG-19..."
	download_archive "VGG-19.tar.gz" "https://nvidia.box.com/shared/static/1ubk73f1akhh4h7mo0iq7erars7j5yyu.gz" 
}

function download_inception_v4()
{
	echo "$LOG Downloading Inception-v4..."
	download_archive "Inception-v4.tar.gz" "https://nvidia.box.com/shared/static/maidbjiwkg6bz2bk7drwq7rj8v4whdl9.gz" 
}

function download_recognition()
{
	echo "$LOG Downloading all Image Recognition models..."

	download_alexnet

	download_googlenet
	download_googlenet12

	download_resnet18
	download_resnet50
	download_resnet101
	download_resnet152

	download_vgg16
	download_vgg19

	download_inception_v4

	ALL_RECOGNITION=1
}


#
# OBJECT DETECTION
#
function download_pednet()
{
	echo "$LOG Downloading PedNet..."
	download_archive "ped-100.tar.gz" "https://nvidia.box.com/shared/static/0wbxo6lmxfamm1dk90l8uewmmbpbcffb.gz" 
}

function download_multiped()
{
	echo "$LOG Downloading MultiPed..."
	download_archive "multiped-500.tar.gz" "https://nvidia.box.com/shared/static/r3bq08qh7zb0ap2lf4ysjujdx64j8ofw.gz" 
}

function download_facenet()
{
	echo "$LOG Downloading FaceNet..."
	download_archive "facenet-120.tar.gz" "https://nvidia.box.com/shared/static/wjitc00ef8j6shjilffibm6r2xxcpigz.gz" 
}

function download_detectnet_coco_dog()
{
	echo "$LOG Downloading DetectNet-COCO-Dog..."
	download_archive "DetectNet-COCO-Dog.tar.gz" "https://nvidia.box.com/shared/static/3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz" 
}

function download_detectnet_coco_chair()
{
	echo "$LOG Downloading DetectNet-COCO-Chair..."
	download_archive "DetectNet-COCO-Chair.tar.gz" "https://nvidia.box.com/shared/static/fq0m0en5mmssiizhs9nxw3xtwgnoltf2.gz" 
}

function download_detectnet_coco_bottle()
{
	echo "$LOG Downloading DetectNet-COCO-Bottle..."
	download_archive "DetectNet-COCO-Bottle.tar.gz" "https://nvidia.box.com/shared/static/8bhm91o9yldpf97dcz5d0welgmjy7ucw.gz" 
}

function download_detectnet_coco_airplane()
{
	echo "$LOG Downloading DetectNet-COCO-Airplane..."
	download_archive "DetectNet-COCO-Airplane.tar.gz" "https://nvidia.box.com/shared/static/xi71hlsht5b0y66loeg73rxfa73q561s.gz" 
}

function download_ssd_mobilenet_v1()
{
	echo "$LOG Downloading SSD-Mobilenet-v1..."
	download_archive "SSD-Mobilenet-v1.tar.gz" "https://nvidia.box.com/shared/static/0pg3xi9opwio65df14rdgrtw40ivbk1o.gz" 
}

function download_ssd_mobilenet_v2()
{
	echo "$LOG Downloading SSD-Mobilenet-v2..."
	download_archive "SSD-Mobilenet-v2.tar.gz" "https://nvidia.box.com/shared/static/jcdewxep8vamzm71zajcovza938lygre.gz" 
}

function download_ssd_inception_v2()
{
	echo "$LOG Downloading SSD-Inception-v2..."
	download_archive "SSD-Inception-v2.tar.gz" "https://nvidia.box.com/shared/static/mjq1cel6r5mdk94yb9o6v4nj8gxzlflr.gz" 
}

function download_detection()
{
	echo "$LOG Downloading all Object Detection models..."

	download_ssd_mobilenet_v1
	download_ssd_mobilenet_v2
	download_ssd_inception_v2

	download_pednet
	download_multiped
	download_facenet

	download_detectnet_coco_dog
	download_detectnet_coco_bottle
	download_detectnet_coco_chair
	download_detectnet_coco_airplane

	ALL_DETECTION=1
}


#
# MONO DEPTH
#
function download_monodepth_fcn_mobilenet()
{
	echo "$LOG Downloading MonoDepth-FCN-Mobilenet..."
	download_archive "MonoDepth-FCN-Mobilenet.tar.gz" "https://nvidia.box.com/shared/static/frgbiqeieaja0o8b0eyb87fjbsqd4zup.gz" 
}

function download_monodepth_fcn_resnet18()
{
	echo "$LOG Downloading MonoDepth-FCN-ResNet18..."
	download_archive "MonoDepth-FCN-ResNet18.tar.gz" "https://nvidia.box.com/shared/static/ai2sxrp1tg8mk4j0jbrw3vthqjp8x0af.gz" 
}

function download_monodepth_fcn_resnet50()
{
	echo "$LOG Downloading MonoDepth-FCN-ResNet50..."
	download_archive "MonoDepth-FCN-ResNet50.tar.gz" "https://nvidia.box.com/shared/static/3umpq9yrv3nj3ltiwlooijx5of414gbh.gz" 
}

function download_monodepth()
{
	echo "$LOG Downloading all Mono Depth models..."

	download_monodepth_fcn_mobilenet
	download_monodepth_fcn_resnet18
	download_monodepth_fcn_resnet50

	ALL_MONO_DEPTH=1
}


#
# POSE ESTIMATION
#
function download_pose_resnet18_body()
{
	echo "$LOG Downloading Pose-ResNet18-Body..."
	download_archive "Pose-ResNet18-Body.tar.gz" "https://nvidia.box.com/shared/static/waf8bsu58v9qh9qj3sp3wsw1nyj61xm5.gz" 
}

function download_pose_resnet18_hand()
{
	echo "$LOG Downloading Pose-ResNet18-Hand..."
	download_archive "Pose-ResNet18-Hand.tar.gz" "https://nvidia.box.com/shared/static/srfcadyqv4eaeq6lvu5qpsm6l5oatcnq.gz" 
}

function download_pose_densenet121_body()
{
	echo "$LOG Downloading Pose-DenseNet121-Body..."
	download_archive "Pose-DenseNet121-Body.tar.gz" "https://nvidia.box.com/shared/static/sizfwdkjmvzlo96serrs7u175wxldrv5.gz" 
}

function download_pose()
{
	echo "$LOG Downloading all Pose Estimation models..."

	download_pose_resnet18_body
	download_pose_resnet18_hand
	download_pose_densenet121_body

	ALL_POSE=1
}


#
# SEMANTIC SEGMENTATION
#
function download_fcn_resnet18_cityscapes_512x256()
{
	echo "$LOG Downloading FCN-ResNet18-Cityscapes-512x256..."
	download_archive "FCN-ResNet18-Cityscapes-512x256.tar.gz" "https://nvidia.box.com/shared/static/k7s7gdgi098309fndm2xbssj553vf71s.gz" 
}

function download_fcn_resnet18_cityscapes_1024x512()
{
	echo "$LOG Downloading FCN-ResNet18-Cityscapes-1024x512..."
	download_archive "FCN-ResNet18-Cityscapes-1024x512.tar.gz" "https://nvidia.box.com/shared/static/9aqg4gpjmk7ipz4z0raa5mvs35om6emy.gz" 
}

function download_fcn_resnet18_cityscapes_2048x1024()
{
	echo "$LOG Downloading FCN-ResNet18-Cityscapes-2048x1024..."
	download_archive "FCN-ResNet18-Cityscapes-2048x1024.tar.gz" "https://nvidia.box.com/shared/static/ylh3d2qk8qvitalq8sy803o7avrb6w0h.gz" 
}

function download_fcn_resnet18_deepscene_576x320()
{
	echo "$LOG Downloading FCN-ResNet18-DeepScene-576x320..."
	download_archive "FCN-ResNet18-DeepScene-576x320.tar.gz" "https://nvidia.box.com/shared/static/jm0zlezvweiimpzluohg6453s0u0nvcv.gz" 
}

function download_fcn_resnet18_deepscene_864x480()
{
	echo "$LOG Downloading FCN-ResNet18-DeepScene-864x480..."
	download_archive "FCN-ResNet18-DeepScene-864x480.tar.gz" "https://nvidia.box.com/shared/static/gooux9b5nknk8wlk60ou9s2unpo760iq.gz" 
}

function download_fcn_resnet18_mhp_512x320()
{
	echo "$LOG Downloading FCN-ResNet18-MHP-512x320..."
	download_archive "FCN-ResNet18-MHP-512x320.tar.gz" "https://nvidia.box.com/shared/static/dgaw0ave3bdws1t5ed333ftx5dbpt9zv.gz" 
}

function download_fcn_resnet18_mhp_640x360()
{
	echo "$LOG Downloading FCN-ResNet18-MHP-640x360..."
	download_archive "FCN-ResNet18-MHP-640x360.tar.gz" "https://nvidia.box.com/shared/static/50mvlrjwbq9ugkmnnqp1sm99g2j21sfn.gz" 
}

function download_fcn_resnet18_pascal_voc_320x320()
{
	echo "$LOG Downloading FCN-ResNet18-Pascal-VOC-320x320..."
	download_archive "FCN-ResNet18-Pascal-VOC-320x320.tar.gz" "https://nvidia.box.com/shared/static/p63pgrr6tm33tn23913gq6qvaiarydaj.gz" 
}

function download_fcn_resnet18_pascal_voc_512x320()
{
	echo "$LOG Downloading FCN-ResNet18-Pascal-VOC-512x320..."
	download_archive "FCN-ResNet18-Pascal-VOC-512x320.tar.gz" "https://nvidia.box.com/shared/static/njup7f3vu4mgju89kfre98olwljws5pk.gz" 
}

function download_fcn_resnet18_sun_rgbd_512x400()
{
	echo "$LOG Downloading FCN-ResNet18-SUN-RGBD-512x400..."
	download_archive "FCN-ResNet18-SUN-RGBD-512x400.tar.gz" "https://nvidia.box.com/shared/static/5vs9t2wah5axav11k8o3l9skb7yy3xgd.gz" 
}

function download_fcn_resnet18_sun_rgbd_640x512()
{
	echo "$LOG Downloading FCN-ResNet18-SUN-RGBD-640x512..."
	download_archive "FCN-ResNet18-SUN-RGBD-640x512.tar.gz" "https://nvidia.box.com/shared/static/z5llxysbcqd8zzzsm7vjqeihs7ihdw20.gz" 
}

function download_segmentation()
{
	echo "$LOG Downloading all Semantic Segmentation models..."

	download_fcn_resnet18_cityscapes_512x256
	download_fcn_resnet18_cityscapes_1024x512
	download_fcn_resnet18_cityscapes_2048x1024
	download_fcn_resnet18_deepscene_576x320
	download_fcn_resnet18_deepscene_864x480
	download_fcn_resnet18_mhp_512x320
	download_fcn_resnet18_mhp_640x360
	download_fcn_resnet18_pascal_voc_320x320
	download_fcn_resnet18_pascal_voc_512x320
	download_fcn_resnet18_sun_rgbd_512x400
	download_fcn_resnet18_sun_rgbd_640x512
	
	ALL_SEGMENTATION=1
}


#
# SEMANTIC SEGMENTATION (legacy)
#
function download_fcn_alexnet_cityscapes_sd()
{
	echo "$LOG Downloading FCN-Alexnet-Cityscapes-SD..."
	download_archive "FCN-Alexnet-Cityscapes-SD.tar.gz" "https://nvidia.box.com/shared/static/pa5d338t9ntca5chfbymnur53aykhall.gz" 
}

function download_fcn_alexnet_cityscapes_hd()
{
	echo "$LOG Downloading FCN-Alexnet-Cityscapes-HD..."
	download_archive "FCN-Alexnet-Cityscapes-HD.tar.gz" "https://nvidia.box.com/shared/static/mh121fvmveemujut7d8c9cbmglq18vz3.gz" 
}

function download_fcn_alexnet_aerial_fpv()
{
	echo "$LOG Downloading FCN-Alexnet-Aerial-FPV..."
	download_archive "FCN-Alexnet-Aerial-FPV-720p.tar.gz" "https://nvidia.box.com/shared/static/y1mzlwkmytzwg2m7akt7tcbsd33f9opz.gz" 
}

function download_fcn_alexnet_pascal_voc()
{
	echo "$LOG Downloading FCN-Alexnet-Pascal-VOC..."
	download_archive "FCN-Alexnet-Pascal-VOC.tar.gz" "https://nvidia.box.com/shared/static/xj20b6qopfwkkpqm12ffiuaekk6bs8op.gz" 
}

function download_fcn_alexnet_synthia_cvpr()
{
	echo "$LOG Downloading FCN-Alexnet-Synthia-CVPR..."
	download_archive "FCN-Alexnet-SYNTHIA-CVPR16.tar.gz" "https://nvidia.box.com/shared/static/u5ey2ws0nbtzyqyftkuqazx1honw6wry.gz" 
}

function download_fcn_alexnet_synthia_summer_sd()
{
	echo "$LOG Downloading FCN-Alexnet-Synthia-Summer-SD..."
	download_archive "FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz" "https://nvidia.box.com/shared/static/vbk5ofu1x2hwp9luanbg4o0vrfub3a7j.gz" 
}

function download_fcn_alexnet_synthia_summer_hd()
{
	echo "$LOG Downloading FCN-Alexnet-Synthia-Summer-HD..."
	download_archive "FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz" "https://nvidia.box.com/shared/static/ydgmqgdhbvul6q9avoc9flxr3fdoa8pw.gz" 
}

function download_segmentation_legacy()
{
	echo "$LOG Downloading all Semantic Segmentation (Legacy) models..."

	download_fcn_alexnet_cityscapes_sd
	download_fcn_alexnet_cityscapes_hd
	download_fcn_alexnet_aerial_fpv
	download_fcn_alexnet_pascal_voc
	download_fcn_alexnet_synthia_cvpr
	download_fcn_alexnet_synthia_summer_sd
	download_fcn_alexnet_synthia_summer_hd

	ALL_SEGMENTATION_LEGACY=1
}


#
# IMAGE PROCESSING
#
function download_deep_homography_coco()
{
	echo "$LOG Downloading Deep-Homography-COCO..."
	download_archive "Deep-Homography-COCO.tar.gz" "https://nvidia.box.com/shared/static/nlqbsdnt76y0nmkwdzxkg4zbvhk4bidh.gz" 
}

function download_super_resolution_bsd500()
{
	echo "$LOG Downloading Super-Resolution-BSD500..."
	download_archive "Super-Resolution-BSD500.tar.gz" "https://nvidia.box.com/shared/static/a99l8ttk21p3tubjbyhfn4gh37o45rn8.gz" 
}

function download_image_processing()
{
	echo "$LOG Downloading all Image Processing models..."

	download_deep_homography_coco
	download_super_resolution_bsd500

	ALL_IMAGE_PROCESSING=1
}


#
# check if a particular deb package is installed with dpkg-query
# arg $1 -> package name
# arg $2 -> variable name to output status to (e.g. HAS_PACKAGE=1)
#
function find_deb_package()
{
	local PKG_NAME=$1
	local HAS_PKG=`dpkg-query -W --showformat='${Status}\n' $PKG_NAME|grep "install ok installed"`

	if [ "$HAS_PKG" == "" ]; then
		echo "$LOG Checking for '$PKG_NAME' deb package...not installed"
	else
		echo "$LOG Checking for '$PKG_NAME' deb package...installed"
		eval "$2=INSTALLED"
	fi
}


#
# install a debian package if it isn't already installed
# arg $1 -> package name
# arg $2 -> variable name to output status to (e.g. FOUND_PACKAGE=INSTALLED)
#
function install_deb_package()
{
	local PKG_NAME=$1
	
	# check to see if the package is already installed
	find_deb_package $PKG_NAME $2

	# if not, install the package
	if [ -z $2 ]; then
		echo "$LOG Missing '$PKG_NAME' deb package...installing '$PKG_NAME' package."
		sudo apt-get --force-yes --yes install $PKG_NAME
	else
		return 0
	fi
	
	# verify that the package was installed
	find_deb_package $PKG_NAME $2
	
	if [ -z $2 ]; then
		echo "$LOG Failed to install '$PKG_NAME' deb package."
		return 1
	else
		echo "$LOG Successfully installed '$PKG_NAME' deb package."
		return 0
	fi
}


#
# non-interactive mode
#
echo "$LOG BUILD_INTERACTVE=$BUILD_INTERACTIVE"

if [[ "$BUILD_INTERACTIVE" != "YES" ]]; then

	echo "$LOG Downloading default models..."

	download_googlenet
	download_resnet18

	download_ssd_mobilenet_v2
	download_pednet
	download_facenet
	download_detectnet_coco_dog

	download_fcn_resnet18_cityscapes_512x256
	download_fcn_resnet18_cityscapes_1024x512
	download_fcn_resnet18_deepscene_576x320
	download_fcn_resnet18_mhp_512x320
	download_fcn_resnet18_pascal_voc_320x320
	download_fcn_resnet18_sun_rgbd_512x400

	exit_message 0
fi


# check for dialog package
install_deb_package "dialog" FOUND_DIALOG
echo "$LOG FOUND_DIALOG=$FOUND_DIALOG"

# use customized RC config
export DIALOGRC=./download-models.rc


#
# main menu
#
while true; do

	models_selected=$(dialog --backtitle "$APP_TITLE" \
							  --title "Model Downloader" \
							  --cancel-label "Quit" \
							  --colors \
							  --checklist "Keys:\n  ↑↓  Navigate Menu\n  Space to Select Models \n  Enter to Continue" 20 80 10 \
							  --output-fd 1 \
							  1 "\ZbImage Recognition - all models  (2.2 GB)\Zn" off \
							  2 "   > AlexNet                    (244 MB)" off \
							  3 "   > GoogleNet                  (54 MB)" on \
							  4 "   > GoogleNet-12               (42 MB)" off \
							  5 "   > ResNet-18                  (47 MB)" on \
							  6 "   > ResNet-50                  (102 MB)" off \
							  7 "   > ResNet-101                 (179 MB)" off \
							  8 "   > ResNet-152                 (242 MB)" off \
							  9 "   > VGG-16                     (554 MB)" off \
							  10 "   > VGG-19                     (575 MB)" off \
							  11 "   > Inception-v4               (172 MB)" off \
							  12 "\ZbObject Detection - all models   (395 MB)\Zn" off \
							  13 "   > SSD-Mobilenet-v1           (27 MB)" off \
							  14 "   > SSD-Mobilenet-v2           (68 MB)" on \
							  15 "   > SSD-Inception-v2           (100 MB)" off \
							  16 "   > PedNet                     (30 MB)" off \
							  17 "   > MultiPed                   (30 MB)" off \
							  18 "   > FaceNet                    (24 MB)" off \
							  19 "   > DetectNet-COCO-Dog         (29 MB)" off \
							  20 "   > DetectNet-COCO-Bottle      (29 MB)" off \
							  21 "   > DetectNet-COCO-Chair       (29 MB)" off \
							  22 "   > DetectNet-COCO-Airplane    (29 MB)" off \
							  23 "\ZbMono Depth - all models         (146 MB)\Zn" off \
							  24 "   > MonoDepth-FCN-Mobilenet    (5 MB)" on \
							  25 "   > MonoDepth-FCN-ResNet18     (40 MB)" off \
							  26 "   > MonoDepth-FCN-ResNet50     (100 MB)" off \
							  27 "\ZbPose Estimation - all models    (222 MB)\Zn" off \
							  28 "   > Pose-ResNet18-Body         (74 MB)" on \
							  29 "   > Pose-ResNet18-Hand         (74 MB)" on \
							  30 "   > Pose-DenseNet121-Body      (74 MB)" off \
							  31 "\ZbSemantic Segmentation - all            (518 MB)\Zn" off \
							  32 "   > FCN-ResNet18-Cityscapes-512x256   (47 MB)" on \
							  33 "   > FCN-ResNet18-Cityscapes-1024x512  (47 MB)" on \
							  34 "   > FCN-ResNet18-Cityscapes-2048x1024 (47 MB)" off \
							  35 "   > FCN-ResNet18-DeepScene-576x320    (47 MB)" on \
							  36 "   > FCN-ResNet18-DeepScene-864x480    (47 MB)" off \
							  37 "   > FCN-ResNet18-MHP-512x320          (47 MB)" on \
							  38 "   > FCN-ResNet18-MHP-640x360          (47 MB)" off \
							  39 "   > FCN-ResNet18-Pascal-VOC-320x320   (47 MB)" on \
							  40 "   > FCN-ResNet18-Pascal-VOC-512x320   (47 MB)" off \
							  41 "   > FCN-ResNet18-SUN-RGBD-512x400     (47 MB)" on \
							  42 "   > FCN-ResNet18-SUN-RGBD-640x512     (47 MB)" off \
							  43 "\ZbSemantic Segmentation - legacy     (1.4 GB)\Zn" off \
							  44 "   > FCN-Alexnet-Cityscapes-SD     (235 MB)" off \
							  45 "   > FCN-Alexnet-Cityscapes-HD     (235 MB)" off \
							  46 "   > FCN-Alexnet-Aerial-FPV        (7 MB)" off \
							  47 "   > FCN-Alexnet-Pascal-VOC        (235 MB)" off \
							  48 "   > FCN-Alexnet-Synthia-CVPR      (235 MB)" off \
							  49 "   > FCN-Alexnet-Synthia-Summer-SD (235 MB)" off \
							  50 "   > FCN-Alexnet-Synthia-Summer-HD (235 MB)" off \
							  51 "\ZbImage Processing - all models   (138 MB)\Zn" off \
							  52 "   > Deep-Homography-COCO       (137 MB)" off \
							  53 "   > Super-Resolution-BSD500    (1 MB)" off )

	model_selection_status=$?
	clear

	echo "$LOG Model selection status:  $model_selection_status"

	if [ $model_selection_status = 0 ]; then

		if [ -z "$models_selected" ]; then
			echo "$LOG No models were selected for download."
		else
			echo "$LOG Models selected for download:  $models_selected"
		
			for model in $models_selected
			do
				if [ $model = 1 ]; then
					download_recognition
				elif [ $model = 2 ] && [ -z $ALL_RECOGNITION ]; then
					download_alexnet
				elif [ $model = 3 ] && [ -z $ALL_RECOGNITION ]; then
					download_googlenet
				elif [ $model = 4 ] && [ -z $ALL_RECOGNITION ]; then
					download_googlenet12
				elif [ $model = 5 ] && [ -z $ALL_RECOGNITION ]; then
					download_resnet18
				elif [ $model = 6 ] && [ -z $ALL_RECOGNITION ]; then
					download_resnet50
				elif [ $model = 7 ] && [ -z $ALL_RECOGNITION ]; then
					download_resnet101
				elif [ $model = 8 ] && [ -z $ALL_RECOGNITION ]; then
					download_resnet152
				elif [ $model = 9 ] && [ -z $ALL_RECOGNITION ]; then
					download_vgg16
				elif [ $model = 10 ] && [ -z $ALL_RECOGNITION ]; then
					download_vgg19
				elif [ $model = 11 ] && [ -z $ALL_RECOGNITION ]; then
					download_inception_v4
				elif [ $model = 12 ]; then
					download_detection
				elif [ $model = 13 ] && [ -z $ALL_DETECTION ]; then
					download_ssd_mobilenet_v1
				elif [ $model = 14 ] && [ -z $ALL_DETECTION ]; then
					download_ssd_mobilenet_v2
				elif [ $model = 15 ] && [ -z $ALL_DETECTION ]; then
					download_ssd_inception_v2
				elif [ $model = 16 ] && [ -z $ALL_DETECTION ]; then
					download_pednet
				elif [ $model = 17 ] && [ -z $ALL_DETECTION ]; then
					download_multiped
				elif [ $model = 18 ] && [ -z $ALL_DETECTION ]; then
					download_facenet
				elif [ $model = 19 ] && [ -z $ALL_DETECTION ]; then
					download_detectnet_coco_dog
				elif [ $model = 20 ] && [ -z $ALL_DETECTION ]; then
					download_detectnet_coco_bottle
				elif [ $model = 21 ] && [ -z $ALL_DETECTION ]; then
					download_detectnet_coco_chair
				elif [ $model = 22 ] && [ -z $ALL_DETECTION ]; then
					download_detectnet_coco_airplane
				elif [ $model = 23 ]; then
					download_monodepth
				elif [ $model = 24 ] && [ -z $ALL_MONO_DEPTH ]; then
					download_monodepth_fcn_mobilenet
				elif [ $model = 25 ] && [ -z $ALL_MONO_DEPTH ]; then
					download_monodepth_fcn_resnet18
				elif [ $model = 26 ] && [ -z $ALL_MONO_DEPTH ]; then
					download_monodepth_fcn_resnet50
				elif [ $model = 27 ]; then
					download_pose
				elif [ $model = 28 ] && [ -z $ALL_POSE ]; then
					download_pose_resnet18_body
				elif [ $model = 29 ] && [ -z $ALL_POSE ]; then
					download_pose_resnet18_hand
				elif [ $model = 30 ] && [ -z $ALL_POSE ]; then
					download_pose_densenet121_body
				elif [ $model = 31 ]; then
					download_segmentation
				elif [ $model = 32 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_cityscapes_512x256
				elif [ $model = 33 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_cityscapes_1024x512
				elif [ $model = 34 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_cityscapes_2048x1024
				elif [ $model = 35 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_deepscene_576x320
				elif [ $model = 36 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_deepscene_864x480
				elif [ $model = 37 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_mhp_512x320
				elif [ $model = 38 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_mhp_640x360
				elif [ $model = 39 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_pascal_voc_320x320
				elif [ $model = 40 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_pascal_voc_512x320
				elif [ $model = 41 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_sun_rgbd_512x400
				elif [ $model = 42 ] && [ -z $ALL_SEGMENTATION ]; then
					download_fcn_resnet18_sun_rgbd_640x512
				elif [ $model = 43 ]; then
					download_segmentation_legacy
				elif [ $model = 44 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_cityscapes_sd
				elif [ $model = 45 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_cityscapes_hd
				elif [ $model = 46 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_aerial_fpv
				elif [ $model = 47 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_pascal_voc
				elif [ $model = 48 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_synthia_cvpr
				elif [ $model = 49 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_synthia_summer_sd
				elif [ $model = 50 ] && [ -z $ALL_SEGMENTATION_LEGACY ]; then
					download_fcn_alexnet_synthia_summer_hd
				elif [ $model = 51 ]; then
					download_image_processing
				elif [ $model = 52 ] && [ -z $ALL_IMAGE_PROCESSING ]; then
					download_deep_homography_coco
				elif [ $model = 53 ] && [ -z $ALL_IMAGE_PROCESSING ]; then
					download_super_resolution_bsd500
				fi
			done
		fi

		exit_message 0
	else
		echo "$LOG Model selection cancelled."
		exit_message 0
	fi

	exit_message 0
done

