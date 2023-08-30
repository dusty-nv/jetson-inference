#!/usr/bin/env bash
#
# This script builds the jetson-inference docker container from source.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#

BASE_IMAGE=$1

# find L4T_VERSION
source docker/tag.sh

if [ -z $BASE_IMAGE ]; then
	if [ $ARCH = "aarch64" ]; then
		if [ $L4T_VERSION = "35.4.1" ]; then
			BASE_IMAGE="dustynv/l4t-pytorch:r35.4.1"
		elif [ $L4T_VERSION = "35.3.1" ]; then
			BASE_IMAGE="dustynv/l4t-pytorch:r35.3.1"
		elif [ $L4T_VERSION = "35.2.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3"
		elif [ $L4T_VERSION = "35.1.0" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.12-py3"
		elif [ $L4T_VERSION = "34.1.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.12-py3"
		elif [ $L4T_VERSION = "34.1.0" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r34.1.0-pth1.12-py3"
		elif [ $L4T_VERSION = "32.7.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.7.1-pth1.10-py3"
		elif [ $L4T_VERSION = "32.6.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.6.1-pth1.9-py3"
		elif [ $L4T_VERSION = "32.5.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3"
		elif [ $L4T_VERSION = "32.5.0" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3"
		elif [ $L4T_VERSION = "32.4.4" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.4-pth1.6-py3"
		elif [ $L4T_VERSION = "32.4.3" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3"
		elif [ $L4T_VERSION = "32.4.2" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-pytorch:r32.4.2-pth1.5-py3"
		else
			echo "cannot automatically select l4t-pytorch base container for L4T R$L4T_VERSION"
			echo "please specify it manually as:  docker/build.sh nvcr.io/nvidia/l4t-pytorch:<TAG>"
			exit 1
		fi
	elif [ $ARCH = "x86_64" ]; then
		BASE_IMAGE="nvcr.io/nvidia/pytorch:$CONTAINER_TAG-py3"
	fi
fi

echo "BASE_IMAGE=$BASE_IMAGE"
echo "CONTAINER_IMAGE=$CONTAINER_LOCAL_IMAGE"

# distro release-dependent build options 
source docker/containers/scripts/opencv_version.sh
		
# build the container
sudo docker build -t $CONTAINER_LOCAL_IMAGE -f Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		--build-arg OPENCV_URL=$OPENCV_URL \
		--build-arg OPENCV_DEB=$OPENCV_DEB \
		.
