#!/usr/bin/env bash
#
# Start the jetson-inference container:
#
#   docker/run.sh [COMMAND] [ARG...]
#
# To run it with ROS, set $ROS_DISTRO environment variable:
#
#   ROS_DISTRO=humble docker/run.sh
#
# Command-line arguments are passed through to 'docker run'
#   https://docs.docker.com/engine/reference/commandline/run/
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

# select $CONTAINER_IMAGE
source $ROOT/docker/tag.sh

# run the container
bash $ROOT/docker/containers/run.sh $DATA_VOLUME $CONTAINER_IMAGE "$@"
