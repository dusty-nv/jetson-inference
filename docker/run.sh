#!/usr/bin/env bash
#
# Start an instance of the jetson-inference docker container.
# See below or run this script with -h or --help to see usage options.
#
# This script should be run from the root dir of the jetson-inference project:
#
#     $ cd /path/to/your/jetson-inference
#     $ docker/run.sh
#

show_help() {
    echo " "
    echo "usage: Starts the Docker container and runs a user-specified command"
    echo " "
    echo "   ./docker/run.sh --container DOCKER_IMAGE"
    echo "                   --volume HOST_DIR:MOUNT_DIR"
    echo "                   --ros ROS_DISTRO"
    echo "                   --run RUN_COMMAND"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   -c, --container DOCKER_IMAGE Specifies the name of the Docker container"
    echo "                                image to use (default: 'jetson-inference')"
    echo " "
    echo "   --dev  Runs the container in development mode, where the source"
    echo "          files are mounted into the container dynamically, so they"
    echo "          can more easily be edited from the host machine."
    echo " "
    echo "   -v, --volume HOST_DIR:MOUNT_DIR Mount a path from the host system into"
    echo "                                   the container.  Should be specified as:"
    echo " "
    echo "                                      -v /my/host/path:/my/container/path"
    echo " "
    echo "                                   These should be absolute paths, and you"
    echo "                                   can specify multiple --volume options."
    echo " "
    echo "  --ros DISTRO Specifies the ROS distro to use, one of:"
    echo "                 'noetic', 'foxy', 'galactic', 'humble', 'iron'"
    echo "               This will enable the use of ros_deep_learning package."
    echo "               When run with just --ros flag, the default distro is foxy."
    echo " "
    echo "   -r, --run RUN_COMMAND  Command to run once the container is started."
    echo "                          Note that this argument must be invoked last,"
    echo "                          as all further arguments will form the command."
    echo "                          If no run command is specified, an interactive"
    echo "                          terminal into the container will be provided."
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

# paths to some project directories
NETWORKS_DIR="data/networks"
CLASSIFY_DIR="python/training/classification"
DETECTION_DIR="python/training/detection/ssd"
RECOGNIZER_DIR="python/www/recognizer"

DOCKER_ROOT="/jetson-inference"	# where the project resides inside docker

# generate mount commands
DATA_VOLUME=" \
--volume $PWD/data:$DOCKER_ROOT/data \
--volume $PWD/$CLASSIFY_DIR/data:$DOCKER_ROOT/$CLASSIFY_DIR/data \
--volume $PWD/$CLASSIFY_DIR/models:$DOCKER_ROOT/$CLASSIFY_DIR/models \
--volume $PWD/$DETECTION_DIR/data:$DOCKER_ROOT/$DETECTION_DIR/data \
--volume $PWD/$DETECTION_DIR/models:$DOCKER_ROOT/$DETECTION_DIR/models \
--volume $PWD/$RECOGNIZER_DIR/data:$DOCKER_ROOT/$RECOGNIZER_DIR/data "

# parse user arguments
USER_COMMAND=""
USER_VOLUME=""
DEV_VOLUME=""
ROS_DISTRO=""

while :; do
    case $1 in
        -h|-\?|--help)
            show_help
            exit
            ;;
        -c|--container)  # takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                CONTAINER_IMAGE=$2
                shift
            else
                die 'ERROR: "--container" requires a non-empty option argument.'
            fi
            ;;
        --container=?*)
            CONTAINER_IMAGE=${1#*=} # delete everything up to "=" and assign the remainder.
            ;;
        --container=)  # handle the case of an empty flag
            die 'ERROR: "--container" requires a non-empty option argument.'
            ;;
        --dev)
            DEV_VOLUME=" --volume $PWD:$DOCKER_ROOT "
            ;;
        -v|--volume)
            if [ "$2" ]; then
                USER_VOLUME="$USER_VOLUME --volume $2 "
                shift
            else
                die 'ERROR: "--volume" requires a non-empty option argument.'
            fi
            ;;
        --volume=?*)
            USER_VOLUME="$USER_VOLUME --volume ${1#*=} "
            ;;
        --volume=)
            die 'ERROR: "--volume" requires a non-empty option argument.'
            ;;
        --ros)
            if [ "$2" ]; then
                ROS_DISTRO=$2
                shift
            else
                ROS_DISTRO="foxy"
            fi
            ;;
        --ros=?*)
            ROS_DISTRO=${1#*=}
            ;;
        --ros=)
            die 'ERROR: "--ros" requires a non-empty option argument.'
            ;;
        -r|--run)
            if [ "$2" ]; then
                shift
                USER_COMMAND=" $@ "
            else
                die 'ERROR: "--run" requires a non-empty option argument.'
            fi
            ;;
        --)
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)   # default case: No more options, so break out of the loop.
            break
    esac

    shift
done

# select container tag (unless specified by user)
if [ -z "$CONTAINER_IMAGE" ]; then
	source docker/tag.sh
else
	source docker/containers/scripts/l4t_version.sh
fi

# check for V4L2 devices
V4L2_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	sudo xhost +si:localuser:root
	DISPLAY_DEVICE=" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix "
fi

# print configuration
print_var() 
{
	if [ -n "${!1}" ]; then                                                # reference var by name - https://stackoverflow.com/a/47768983
		local trimmed="$(echo -e "${!1}" | sed -e 's/^[[:space:]]*//')"   # remove leading whitespace - https://stackoverflow.com/a/3232433    
		printf '%-17s %s\n' "$1:" "$trimmed"                              # justify prefix - https://unix.stackexchange.com/a/354094
	fi
}

print_var "CONTAINER_IMAGE"
print_var "ROS_DISTRO"
print_var "DATA_VOLUME"
print_var "DEV_VOLUME"
print_var "USER_VOLUME"
print_var "USER_COMMAND"
print_var "V4L2_DEVICES"
print_var "DISPLAY_DEVICE"

# run the container
if [ $ARCH = "aarch64" ]; then

	# /proc or /sys files aren't mountable into docker
	cat /proc/device-tree/model > /tmp/nv_jetson_model

	sudo docker run --runtime nvidia -it --rm \
		--network host \
		-v /tmp/argus_socket:/tmp/argus_socket \
		-v /etc/enctune.conf:/etc/enctune.conf \
		-v /etc/nv_tegra_release:/etc/nv_tegra_release \
		-v /tmp/nv_jetson_model:/tmp/nv_jetson_model \
		$DISPLAY_DEVICE $V4L2_DEVICES \
		$DATA_VOLUME $USER_VOLUME $DEV_VOLUME \
		$CONTAINER_IMAGE $USER_COMMAND

elif [ $ARCH = "x86_64" ]; then

	sudo docker run --gpus all -it --rm \
		--network=host \
		--shm-size=8g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		$DISPLAY_DEVICE $V4L2_DEVICES \
		$DATA_VOLUME $USER_VOLUME $DEV_VOLUME \
		$CONTAINER_IMAGE $USER_COMMAND
		
fi

