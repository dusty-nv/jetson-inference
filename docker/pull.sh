#!/usr/bin/env bash
#
# This script pulls the jetson-inference docker container image.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/pull.sh
#

# find L4T_VERSION
source tools/l4t-version.sh

CONTAINER_IMAGE="jetson-inference:r$L4T_VERSION"

echo "pulling $CONTAINER_IMAGE"
sudo docker pull $CONTAINER_IMAGE
