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

WORKDIR jetson-inference


#
# install pre-requisite packages
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            cmake \
    && rm -rf /var/lib/apt/lists/*
    
# pip dependencies for pytorch-ssd
RUN pip3 install --verbose boto3 pandas

# alias python3 -> python
RUN rm /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip
    
    
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
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j$(nproc) && \
    make install && \
    /bin/bash -O extglob -c "cd /jetson-inference/build; rm -rf -v !(aarch64|download-models.*)"
    rm -rf /var/lib/apt/lists/*
    