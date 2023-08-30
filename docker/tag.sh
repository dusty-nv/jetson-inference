#!/usr/bin/env bash

# find L4T_VERSION
source docker/containers/scripts/l4t_version.sh

if [ $ARCH = "aarch64" ]; then
	# local container tag
	CONTAINER_TAG="r$L4T_VERSION"
	
	# incompatible L4T version
	version_error()
	{
		echo "Docker containers aren't supported on Jetson prior to JetPack 4.4 / L4T R32.4.3"
		echo "Please upgrade to the latest JetPack, or build jetson-inference natively from source"
		exit 1
	}

	# adjust tags for compatible L4T versions
	if [ $L4T_RELEASE -eq 32 ]; then
		if [[ $L4T_REVISION_MAJOR -lt 4 && $L4T_REVISION_MINOR -gt 4 ]]; then
			# L4T R32.4 was the first version containers are supported on
			version_error
		elif [ $L4T_REVISION_MAJOR -eq 5 ]; then
			# L4T R32.5.x all run the R32.5.0 container
			CONTAINER_TAG="r32.5.0"
		elif [ $L4T_REVISION_MAJOR -eq 7 ]; then
			# L4T R32.7.x all run the R32.7.0 container
			CONTAINER_TAG="r32.7.1"
		fi
	elif [ $L4T_RELEASE -eq 35 ]; then
		if [ $L4T_REVISION_MAJOR -gt 4 ]; then
			CONTAINER_TAG="r35.4.1"
		fi
	fi

elif [ $ARCH = "x86_64" ]; then
	CONTAINER_TAG="22.06"  # NGC pytorch base container tag (TODO: query the latest release)
fi
	
# check if ROS is to be used
if [ -n "$ROS_DISTRO" ]; then
	CONTAINER_IMAGE="ros:$ROS_DISTRO-pytorch-l4t-$CONTAINER_TAG"
else
	CONTAINER_IMAGE="jetson-inference:$CONTAINER_TAG"
fi

# CONTAINER_LOCAL_IMAGE is always the local container name
# CONTAINER_REMOTE_IMAGE is always the container name in NGC/dockerhub
# CONTAINER_IMAGE is either local container name (if it exists on the system) or the remote image name
CONTAINER_LOCAL_IMAGE="$CONTAINER_IMAGE"
CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_LOCAL_IMAGE"

if [[ "$(sudo docker images -q $CONTAINER_LOCAL_IMAGE 2> /dev/null)" == "" ]]; then
	CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
fi
