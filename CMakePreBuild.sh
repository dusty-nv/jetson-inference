#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
TORCH_PREFIX=$PWD/torch

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:       $BUILD_ROOT"


# break on errors
#set -e


# install packages
sudo apt-get update
sudo apt-get install -y libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev libsdl2-dev libsdl2-ttf-dev
# libgstreamer0.10-0-dev libgstreamer-plugins-base0.10-dev libxml2-dev
sudo apt-get update


# libgstreamer-plugins-base1.0-dev

#sudo rm /usr/lib/aarch64-linux-gnu/libGL.so
#sudo ln -s /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so

wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
mv bvlc_alexnet.caffemodel ../data/networks

wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
mv bvlc_googlenet.caffemodel ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOMzVRODNuSHlvbms' -O ped-100.tar.gz
tar -xzvf ped-100.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOUmtGdGIyYjlEbTA' -O multiped-500.tar.gz
tar -xzvf multiped-500.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOWXpOQ0JCQ3AxSTA' -O facenet-120.tar.gz
tar -xzvf facenet-120.tar.gz -C ../data/networks

echo "[Pre-build]  Finished CMakePreBuild script"
