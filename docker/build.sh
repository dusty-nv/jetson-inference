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
source tools/l4t-version.sh

if [ -z $BASE_IMAGE ]; then
	if [ $L4T_VERSION = "32.6.1" ]; then
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
		echo "cannot build jetson-inference docker container for L4T R$L4T_VERSION"
		echo "please upgrade to the latest JetPack, or build jetson-inference natively"
		exit 1
	fi
fi

echo "BASE_IMAGE=$BASE_IMAGE"
echo "TAG=jetson-inference:r$L4T_VERSION"


# sanitize workspace (so extra files aren't added to the container)
rm -rf python/training/classification/data/*
rm -rf python/training/classification/models/*

rm -rf python/training/detection/ssd/data/*
rm -rf python/training/detection/ssd/models/*


# opencv.csv mounts files that preclude us installing different version of opencv
# temporarily disable the opencv.csv mounts while we build the container
CV_CSV="/etc/nvidia-container-runtime/host-files-for-container.d/opencv.csv"

if [ -f "$CV_CSV" ]; then
	sudo mv $CV_CSV $CV_CSV.backup
fi
	
	
# build the container
sudo docker build -t jetson-inference:r$L4T_VERSION -f Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		.


# restore opencv.csv mounts
if [ -f "$CV_CSV.backup" ]; then
	sudo mv $CV_CSV.backup $CV_CSV
fi
