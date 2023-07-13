# 
# name: jetson-inference
# depends: [pytorch, torchvision, gstreamer, opencv]
# notes:  see docs/aux-docker.md for instructions to build
#
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /jetson-inference

  
# 
# install python packages
#
COPY python/training/detection/ssd/requirements.txt /tmp/pytorch_ssd_requirements.txt
COPY python/www/flask/requirements.txt /tmp/flask_requirements.txt
COPY python/www/dash/requirements.txt /tmp/dash_requirements.txt

RUN pip3 install --no-cache-dir --verbose --upgrade Cython && \
    pip3 install --no-cache-dir --verbose -r /tmp/pytorch_ssd_requirements.txt && \
    pip3 install --no-cache-dir --verbose -r /tmp/flask_requirements.txt && \
    pip3 install --no-cache-dir --verbose -r /tmp/dash_requirements.txt
    

#
# copy source
#
COPY c c
COPY examples examples
COPY python python
COPY tools tools
COPY utils utils
COPY data/networks/models.json data/networks/models.json

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
    /bin/bash -O extglob -c "cd /jetson-inference/build; rm -rf -v !($(uname -m)|download-models.*)" && \
    rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
# build out-of-tree samples
RUN cd examples/my-recognition && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make

# workaround for "cannot allocate memory in static TLS block"
ENV LD_PRELOAD=${LD_PRELOAD}:/usr/lib/aarch64-linux-gnu/libgomp.so.1:/lib/aarch64-linux-gnu/libGLdispatch.so.0

# make sure it loads
RUN python3 -c 'import jetson_inference' && \
    python3 -c 'import jetson_utils'
