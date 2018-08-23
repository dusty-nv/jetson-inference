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
if [ ! -f bvlc_alexnet.caffemodel ]; then
    #wget http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/5j264j7mky11q8emy4q14w3r8hl5v6zh.caffemodel' -O bvlc_alexnet.caffemodel.tmp
    mv bvlc_alexnet.caffemodel.tmp bvlc_alexnet.caffemodel
fi
cp bvlc_alexnet.caffemodel ../data/networks

if [ ! -f alexnet.prototxt ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/c84wp3axbtv4e2gybn40jprdquav9azm.prototxt' -O alexnet.prototxt.tmp
    mv alexnet.prototxt.tmp alexnet.prototxt
fi
cp alexnet.prototxt ../data/networks

# GoogleNet (bvlc site was behaving slowly, so enabled mirror on nvidia.box.com instead)
if [ ! -f bvlc_googlenet.caffemodel ]; then
    #wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/at8b1105ww1c5h7p30j5ko8qfnxrs0eg.caffemodel' -O bvlc_googlenet.caffemodel.tmp
    mv bvlc_googlenet.caffemodel.tmp bvlc_googlenet.caffemodel
fi
cp bvlc_googlenet.caffemodel ../data/networks

if [ ! -f googlenet.prototxt ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/5z3l76p8ap4n0o6rk7lyasdog9f14gc7.prototxt' -O googlenet.prototxt.tmp
    mv googlenet.prototxt.tmp googlenet.prototxt
fi
cp googlenet.prototxt ../data/networks

# GoogleNet, ILSVR12 subset
if [ ! -f GoogleNet-ILSVRC12-subset.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/zb8i3zcg39sdjjxfty7o5935hpbd64y4.gz' -O GoogleNet-ILSVRC12-subset.tar.gz.tmp
    mv GoogleNet-ILSVRC12-subset.tar.gz.tmp GoogleNet-ILSVRC12-subset.tar.gz
fi
tar -xzvf GoogleNet-ILSVRC12-subset.tar.gz -C ../data/networks

# DetectNet's  (uncomment to download)
#if [ ! -f detectnet.prototxt ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/xe6wo1o8qiqykfx8umuu0ki9idp0f92p.prototxt' -O detectnet.prototxt.tmp
    #mv detectnet.prototxt.tmp detectnet.prototxt
#fi
cp detectnet.prototxt ../data/networks

if [ ! -f ped-100.tar.gz ]; then
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOMzVRODNuSHlvbms' -O ped-100.tar.gz.tmp
    mv ped-100.tar.gz.tmp ped-100.tar.gz
fi    
tar -xzvf ped-100.tar.gz -C ../data/networks

if [ ! multiped-500.tar.gz ]; then
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOUmtGdGIyYjlEbTA' -O multiped-500.tar.gz.tmp
    mv multiped-500.tar.gz.tmp multiped-500.tar.gz
fi
tar -xzvf multiped-500.tar.gz -C ../data/networks

if [ ! facenet-120.tar.gz ]; then
    wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=0BwYxpotGWRNOWXpOQ0JCQ3AxSTA' -O facenet-120.tar.gz.tmp
    mv facenet-120.tar.gz.tmp facenet-120.tar.gz
fi
tar -xzvf facenet-120.tar.gz -C ../data/networks

# COCO (DetectNet)
if [ ! DetectNet-COCO-Airplane.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/xi71hlsht5b0y66loeg73rxfa73q561s.gz' -O DetectNet-COCO-Airplane.tar.gz.tmp
    mv DetectNet-COCO-Airplane.tar.gz.tmp DetectNet-COCO-Airplane.tar.gz
fi
tar -xzvf DetectNet-COCO-Airplane.tar.gz -C ../data/networks

if [ ! DetectNet-COCO-Bottle.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/8bhm91o9yldpf97dcz5d0welgmjy7ucw.gz' -O DetectNet-COCO-Bottle.tar.gz.tmp
    mv DetectNet-COCO-Bottle.tar.gz.tmp DetectNet-COCO-Bottle.tar.gz
fi
tar -xzvf DetectNet-COCO-Bottle.tar.gz -C ../data/networks

if [ ! DetectNet-COCO-Chair.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/fq0m0en5mmssiizhs9nxw3xtwgnoltf2.gz' -O DetectNet-COCO-Chair.tar.gz.tmp
    mv DetectNet-COCO-Chair.tar.gz.tmp DetectNet-COCO-Chair.tar.gz
fi
tar -xzvf DetectNet-COCO-Chair.tar.gz -C ../data/networks

if [ ! DetectNet-COCO-Dog.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/3qdg3z5qvl8iwjlds6bw7bwi2laloytu.gz' -O DetectNet-COCO-Dog.tar.gz.tmp
    mv DetectNet-COCO-Dog.tar.gz.tmp DetectNet-COCO-Dog.tar.gz
fi
tar -xzvf DetectNet-COCO-Dog.tar.gz -C ../data/networks

# Segmentation Nets (uncomment to download)
if [ ! FCN-Alexnet-Pascal-VOC.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/xj20b6qopfwkkpqm12ffiuaekk6bs8op.gz' -O FCN-Alexnet-Pascal-VOC.tar.gz.tmp
    mv FCN-Alexnet-Pascal-VOC.tar.gz.tmp FCN-Alexnet-Pascal-VOC.tar.gz
fi
tar -xzvf FCN-Alexnet-Pascal-VOC.tar.gz -C ../data/networks

#if [ ! FCN-Alexnet-SYNTHIA-CVPR16.tar.gz ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/u5ey2ws0nbtzyqyftkuqazx1honw6wry.gz' -O FCN-Alexnet-SYNTHIA-CVPR16.tar.gz.tmp
    #mv FCN-Alexnet-SYNTHIA-CVPR16.tar.gz.tmp FCN-Alexnet-SYNTHIA-CVPR16.tar.gz
#fi
tar -xzvf FCN-Alexnet-SYNTHIA-CVPR16.tar.gz -C ../data/networks

#if [ ! FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/ydgmqgdhbvul6q9avoc9flxr3fdoa8pw.gz' -O FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz.tmp
    #mv FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz.tmp FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz
#fi
tar -xzvf FCN-Alexnet-SYNTHIA-Summer-HD.tar.gz -C ../data/networks

#if [ ! FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/vbk5ofu1x2hwp9luanbg4o0vrfub3a7j.gz' -O FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz.tmp
    #mv FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz.tmp FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz
#fi
tar -xzvf FCN-Alexnet-SYNTHIA-Summer-SD.tar.gz -C ../data/networks

if [ ! FCN-Alexnet-Cityscapes-HD.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/mh121fvmveemujut7d8c9cbmglq18vz3.gz' -O FCN-Alexnet-Cityscapes-HD.tar.gz.tmp
    mv FCN-Alexnet-Cityscapes-HD.tar.tmp FCN-Alexnet-Cityscapes-HD.tar
fi
tar -xzvf FCN-Alexnet-Cityscapes-HD.tar.gz -C ../data/networks

#if [ ! FCN-Alexnet-Cityscapes-SD.tar.gz ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/pa5d338t9ntca5chfbymnur53aykhall.gz' -O FCN-Alexnet-Cityscapes-SD.tar.gz.tmp
    #mv FCN-Alexnet-Cityscapes-SD.tar.gz.tmp FCN-Alexnet-Cityscapes-SD.tar.gz
#fi
tar -xzvf FCN-Alexnet-Cityscapes-SD.tar.gz -C ../data/networks

if [ ! FCN-Alexnet-Aerial-FPV-720p.tar.gz ]; then
    wget --no-check-certificate 'https://nvidia.box.com/shared/static/y1mzlwkmytzwg2m7akt7tcbsd33f9opz.gz' -O FCN-Alexnet-Aerial-FPV-720p.tar.gz.tmp
    mv FCN-Alexnet-Aerial-FPV-720p.tar.gz.tmp FCN-Alexnet-Aerial-FPV-720p.tar.gz
fi
tar -xzvf FCN-Alexnet-Aerial-FPV-720p.tar.gz -C ../data/networks

#if [ ! FCN-Alexnet-Aerial-FPV-4ch-720p.tar.gz ]; then
    #wget --no-check-certificate 'https://nvidia.box.com/shared/static/4z5lmlja13blj3mdn6vesrft57p30446.gz' -O FCN-Alexnet-Aerial-FPV-4ch-720p.tar.gz.tmp
    #mv FCN-Alexnet-Aerial-FPV-4ch-720p.tar.gz.tmp FCN-Alexnet-Aerial-FPV-4ch-720p.tar.gz
#fi
tar -xzvf FCN-Alexnet-Aerial-FPV-4ch-720p.tar.gz -C ../data/networks


echo "[Pre-build]  Finished CMakePreBuild script"
