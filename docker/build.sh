#!/usr/bin/env bash
# To build the jetson-inference docker container:
#
#     $ cd /path/to/your/jetson-inference
#     $ pip3 install -r docker/containers/requirements.txt
#     $ docker/build.sh
#
# Also you should set your docker default-runtime to nvidia:
#     https://github.com/dusty-nv/jetson-containers#docker-default-runtime
#
# See https://github.com/dusty-nv/jetson-containers for more build options.
#
ROOT="$( dirname $( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd ) )"

if [ $# -gt 0 ]; then
	bash $ROOT/docker/containers/build.sh --package-dirs=$ROOT "$@"
else
	bash $ROOT/docker/containers/build.sh --package-dirs=$ROOT --skip-tests=intermediate jetson-inference
fi
