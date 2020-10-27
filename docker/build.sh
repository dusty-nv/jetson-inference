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
	if [ $L4T_VERSION = "32.4.4" ]; then
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


# build the container
sudo docker build -t jetson-inference:r$L4T_VERSION -f Dockerfile \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		.

