# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Build this Dockerfile by running the following commands:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#

ARG BASE_IMAGE=nvcr.io/nvidia/l4t-pytorch:r32.4.3-pth1.6-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash

WORKDIR /jetson-inference

        
#
# install development packages
#
RUN apt-get update && \
    apt-get purge -y '*opencv*' || echo "existing OpenCV installation not found" && \
    apt-get install -y --no-install-recommends \
            cmake \
		  nano \
		  mesa-utils \
		  lsb-release \
		  gstreamer1.0-tools \
		  gstreamer1.0-libav \
		  gstreamer1.0-rtsp \
		  gstreamer1.0-plugins-rtp \
		  gstreamer1.0-plugins-good \
		  gstreamer1.0-plugins-bad \
		  gstreamer1.0-plugins-ugly \
		  libgstreamer-plugins-base1.0-dev \
		  libgstreamer-plugins-good1.0-dev \
		  libgstreamer-plugins-bad1.0-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# pip dependencies for pytorch-ssd
RUN pip3 install --verbose --upgrade Cython && \
    pip3 install --verbose boto3 pandas tensorboard

# make a copy of this cause it gets purged...
RUN mkdir -p /usr/local/include/gstreamer-1.0/gst && \
    cp -r /usr/include/gstreamer-1.0/gst/webrtc /usr/local/include/gstreamer-1.0/gst && \
    ls -ll /usr/local/include/ && \
    ls -ll /usr/local/include/gstreamer-1.0/gst/webrtc


# 
# install OpenCV (with CUDA)
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

COPY docker/containers/scripts/opencv_install.sh /tmp/opencv_install.sh
RUN cd /tmp && ./opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}

  
#
# copy source
#
COPY c c
COPY calibration calibration
COPY examples examples
COPY plugins plugins
COPY python python
COPY tools tools
COPY utils utils

COPY CMakeLists.txt CMakeLists.txt
COPY CMakePreBuild.sh CMakePreBuild.sh


#
# build source
#
RUN mkdir docs && \
    touch docs/CMakeLists.txt && \
    sed -i 's/nvcaffe_parser/nvparsers/g' CMakeLists.txt && \
    cp -r /usr/local/include/gstreamer-1.0/gst/webrtc /usr/include/gstreamer-1.0/gst && \
    ln -s /usr/lib/aarch64-linux-gnu/libgstwebrtc-1.0.so.0 /usr/lib/aarch64-linux-gnu/libgstwebrtc-1.0.so && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install && \
    /bin/bash -O extglob -c "cd /jetson-inference/build; rm -rf -v !(aarch64|download-models.*)" && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
