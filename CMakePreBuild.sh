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
sudo apt-get install -y libqt4-dev qt4-dev-tools libglew-dev glew-utils libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libglib2.0-dev
# libgstreamer0.10-0-dev libgstreamer-plugins-base0.10-dev libxml2-dev
sudo apt-get update


# libgstreamer-plugins-base1.0-dev

sudo rm /usr/lib/aarch64-linux-gnu/libGL.so
sudo ln -s /usr/lib/aarch64-linux-gnu/tegra/libGL.so /usr/lib/aarch64-linux-gnu/libGL.so


# uncomment to download Alexnet (220MB)
#wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
#mv bvlc_alexnet.caffemodel ../data/networks

# GoogleNet
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
mv bvlc_googlenet.caffemodel ../data/networks


# DetectNet's  (uncomment to download)
#wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOMzVRODNuSHlvbms' -O ped-100.tar.gz
#tar -xzvf ped-100.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOUmtGdGIyYjlEbTA' -O multiped-500.tar.gz
tar -xzvf multiped-500.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOWXpOQ0JCQ3AxSTA' -O facenet-120.tar.gz
tar -xzvf facenet-120.tar.gz -C ../data/networks


# Segmentation Nets (202MB each - uncomment to download)
#wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOSkJKR2VPalp1WEE' -O FCN-Alexnet-PASCAL-VOC.tar.gz
#tar -xzvf FCN-Alexnet-PASCAL-VOC.tar.gz -C ../data/networks

#wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOVFpIY1JHZzBZcUE' -O FCN-Alexnet-SYNTHIA-CVPR16.tar.gz
#tar -xzvf FCN-Alexnet-SYNTHIA-CVPR16.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOZEtfYVVHLW9FSUU' -O FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz
tar -xzvf FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOZEJVSS1QR1Joemc' -O FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz
tar -xzvf FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOb3MtX0s1RENlTFE' -O FCN-Alexnet-Cityscapes-HD.tar.gz
tar -xzvf FCN-Alexnet-Cityscapes-HD.tar.gz -C ../data/networks

wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOSFllN2RYS0VLeHM' -O FCN-Alexnet-Cityscapes-SD.tar.gz
tar -xzvf FCN-Alexnet-Cityscapes-SD.tar.gz -C ../data/networks


echo "[Pre-build]  Finished CMakePreBuild script"
