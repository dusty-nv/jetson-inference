#!/usr/bin/env bash

# find L4T_VERSION
source docker/containers/scripts/l4t_version.sh

if [ $ARCH = "aarch64" ]; then
	# local container:tag name
	CONTAINER_IMAGE="jetson-inference:r$L4T_VERSION"
	
	# incompatible L4T version
	function version_error()
	{
		echo "cannot find compatible jetson-inference docker container for L4T R$L4T_VERSION"
		echo "please upgrade to the latest JetPack, or build jetson-inference natively from source"
		exit 1
	}

	# get remote container URL
	if [ $L4T_RELEASE -eq 32 ]; then
		if [[ $L4T_REVISION_MAJOR -lt 4 && $L4T_REVISION_MINOR -gt 4 ]]; then
			# L4T R32.4 was the first version containers are supported on
			version_error
		elif [ $L4T_REVISION_MAJOR -eq 5 ]; then
			# L4T R32.5.x all run the R32.5.0 container
			CONTAINER_IMAGE="jetson-inference:r32.5.0"
		elif [ $L4T_REVISION_MAJOR -eq 7 ]; then
			# L4T R32.7.x all run the R32.7.0 container
			CONTAINER_IMAGE="jetson-inference:r32.7.1"
		fi
	fi
	
	CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
	
elif [ $ARCH = "x86_64" ]; then
	# TODO:  add logic here for getting the latest release
	CONTAINER_IMAGE="jetson-inference:22.06"
	CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
fi
	
# TAG is always the local container name
# CONTAINER_REMOTE_IMAGE is always the container name in NGC/dockerhub
# CONTAINER_IMAGE is either local container name (if it exists on the system) or the remote image name
TAG=$CONTAINER_IMAGE

# check for local image
if [[ "$(sudo docker images -q $CONTAINER_IMAGE 2> /dev/null)" == "" ]]; then
	CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
fi
