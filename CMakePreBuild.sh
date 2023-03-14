#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
BUILD_INTERACTIVE=$1

if [ -f /.dockerenv ]; then
	BUILD_CONTAINER="YES"
else
	BUILD_CONTAINER="NO"
fi

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory: $BUILD_ROOT"
echo "[Pre-build]  build interactive:    $BUILD_INTERACTIVE"
echo "[Pre-build]  build container:      $BUILD_CONTAINER"
echo " "

# detect build architecture
ARCH=$(uname -i)

# break on errors
#set -e

# docker doesn't use sudo
if [ $BUILD_CONTAINER = "YES" ]; then
	SUDO=""
else
	SUDO="sudo"
fi
	
# install packages
$SUDO apt-get update
$SUDO apt-get install -y --no-install-recommends \
		dialog \
		libglew-dev \
		glew-utils \
		gstreamer1.0-libav \
		gstreamer1.0-nice \
		libgstreamer1.0-dev \
		libgstrtspserver-1.0-dev \
		libglib2.0-dev \
		libsoup2.4-dev \
		libjson-glib-dev \
		python3-pip \
		python3-packaging \
		qtbase5-dev

if [ $BUILD_CONTAINER = "NO" ]; then
	# these are installed in a different step in the Dockerfile
	$SUDO apt-get install -y --no-install-recommends \
		libgstreamer-plugins-base1.0-dev \
		libgstreamer-plugins-good1.0-dev \
		libgstreamer-plugins-bad1.0-dev
fi

if [ $ARCH != "x86_64" ]; then
	# on x86, these are already installed by conda and installing them again creates conflicts
	$SUDO apt-get install -y libpython3-dev python3-numpy
fi

# install cython for if numpy gets built by later packages
pip3 install --no-cache-dir --verbose --upgrade Cython

# download/install models and PyTorch
if [ $BUILD_CONTAINER = "NO" ]; then
	#./download-models.sh $BUILD_INTERACTIVE
	./install-pytorch.sh $BUILD_INTERACTIVE
fi

echo "[Pre-build]  Finished CMakePreBuild script"
