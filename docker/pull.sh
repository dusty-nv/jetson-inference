#!/usr/bin/env bash
#
# This script pulls the jetson-inference docker container image.
# It should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/pull.sh
#

# find container tag from L4T version
source docker/tag.sh

echo "pulling $CONTAINER_IMAGE"
sudo docker pull $CONTAINER_IMAGE
