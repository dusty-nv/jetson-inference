#!/usr/bin/env bash

# find L4T_VERSION
source tools/l4t-version.sh

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
	if [ $L4T_REVISION_MAJOR -eq 4 ]; then
	     if [ $L4T_REVISION_MINOR -gt 4 ]; then
			CONTAINER_REMOTE_IMAGE="nvcr.io/ea-linux4tegra/$CONTAINER_IMAGE"
		elif [ $L4T_REVISION_MINOR -ge 3 ]; then
			CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
		else
			version_error
		fi
	elif [ $L4T_REVISION_MAJOR -eq 5 ]; then
		if [ $L4T_REVISION_MINOR -eq 0 ]; then
			CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
	     elif [ $L4T_REVISION_MINOR -eq 1 ] || [ $L4T_REVISION_MINOR -eq 2 ]; then
			# L4T R32.5.1 / R32.5.2 runs the R32.5.0 container
			CONTAINER_IMAGE="jetson-inference:r32.5.0"
			CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
		else
			CONTAINER_REMOTE_IMAGE="nvcr.io/ea-linux4tegra/$CONTAINER_IMAGE"
		fi
	elif [ $L4T_REVISION_MAJOR -gt 5 ]; then
		CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"
	else
		version_error
	fi
else
	version_error
fi
	
# check for local image
if [[ "$(sudo docker images -q $CONTAINER_IMAGE 2> /dev/null)" == "" ]]; then
	CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
fi


