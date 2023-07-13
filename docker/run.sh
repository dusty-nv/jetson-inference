#!/usr/bin/env bash
#
# To start the container, for example from the project root:
#
#   docker/run.sh
#
# To run container with ROS included, set $ROS_DISTRO environment variable:
#
#   ROS_DISTRO=humble docker/run.sh
#
# When no command-line arguments are given, the container image to run 
# will automatically be selected by docker/tag.sh (in its $IMAGE var),
# including pulling it from DockerHub if it hasn't been built locally.
#
# To specify arguments to pass through to 'docker run', you must also 
# specify the container image (and can use docker/tag.sh if desired)
# 
#   source docker/tag.sh
#   docker/run.sh --name xyz --volume my_dir:/mount $IMAGE /bin/bash
#
# -or-
#
#   IMAGE=$(docker/tag.sh) docker/run.sh --name xyz $IMAGE /bin/bash
#
# Args:  https://docs.docker.com/engine/reference/commandline/run/ 
#
ROOT="$( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd ) )"

# paths to some project directories
NETWORKS_DIR="data/networks"
CLASSIFY_DIR="python/training/classification"
DETECTION_DIR="python/training/detection/ssd"
RECOGNIZER_DIR="python/www/recognizer"

DOCKER_ROOT="/jetson-inference"	# where the project resides inside docker

# generate mount commands
DATA_VOLUME=" \
--volume $ROOT/data:$DOCKER_ROOT/data \
--volume $ROOT/$CLASSIFY_DIR/data:$DOCKER_ROOT/$CLASSIFY_DIR/data \
--volume $ROOT/$CLASSIFY_DIR/models:$DOCKER_ROOT/$CLASSIFY_DIR/models \
--volume $ROOT/$DETECTION_DIR/data:$DOCKER_ROOT/$DETECTION_DIR/data \
--volume $ROOT/$DETECTION_DIR/models:$DOCKER_ROOT/$DETECTION_DIR/models \
--volume $ROOT/$RECOGNIZER_DIR/data:$DOCKER_ROOT/$RECOGNIZER_DIR/data "

# select image if no container was specified 
if [ $# -eq 0 ]; then
	source $ROOT/docker/tag.sh
fi

# run the container
bash $ROOT/docker/containers/run.sh $DATA_VOLUME $IMAGE "$@" 
