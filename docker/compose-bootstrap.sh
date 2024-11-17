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
    echo "usage: Bootstrap the enironment required for running docker-compose"
    echo " "
    echo "   ./docker/compose-bootstrap.sh"
    echo " "
    echo "After this, edit docker-compose.yml for correct parameters, and then run ./docker/docker-compose up -d to launch container"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

if [[ -x "$PWD/docker/docker-compose" ]]; then
    echo "Latest docker-compose configured already, skipping download."
else
    echo "Installing latest docker-compose to ./docker/docker-compose. Assuming that curl is installed."
    # Prevent conflict with local docker-compose installation by maintaing our own binary. It is simple since it is Go static-linked binary anyway.
    curl -L "https://github.com/docker/compose/releases/download/v2.1.1/docker-compose-$(uname -s)-$(uname -m)" -o $PWD/docker/docker-compose
    chmod +x $PWD/docker/docker-compose
fi

# find container tag from L4T version
source docker/tag.sh

# paths to some project directories
NETWORKS_DIR="data/networks"
CLASSIFY_DIR="python/training/classification"
DETECTION_DIR="python/training/detection/ssd"

DOCKER_ROOT="/jetson-inference"	# where the project resides inside docker

# check if we need to download models
SIZE_MODELS=$(du -sb $NETWORKS_DIR | cut -f 1)  

echo "size of $NETWORKS_DIR:  $SIZE_MODELS bytes"
  
if [[ $SIZE_MODELS -lt 204800 ]]; then  # some text files come with the repo (~78KB), so check for a bit more than that
	sudo apt-get update
	sudo apt-get install dialog
	echo "Models have not yet been downloaded, running model downloader tool now..."
	cd tools
	./download-models.sh
	cd ../
fi

# check for pytorch-ssd base model
SSD_BASE_MODEL="$DETECTION_DIR/models/mobilenet-v1-ssd-mp-0_675.pth"

if [ ! -f "$SSD_BASE_MODEL" ]; then
	echo "Downloading pytorch-ssd base model..."
	wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate https://nvidia.box.com/shared/static/djf5w54rjvpqocsiztzaandq1m3avr7c.pth -O $SSD_BASE_MODEL
fi

mkdir -p $PWD/$DETECTION_DIR/code

# generate mount commands
DATA_VOLUME="\
--volume $PWD/data:$DOCKER_ROOT/data \
--volume $PWD/$CLASSIFY_DIR/data:$DOCKER_ROOT/$CLASSIFY_DIR/data \
--volume $PWD/$CLASSIFY_DIR/models:$DOCKER_ROOT/$CLASSIFY_DIR/models \
--volume $PWD/$DETECTION_DIR/data:$DOCKER_ROOT/$DETECTION_DIR/data \
--volume $PWD/$DETECTION_DIR/models:$DOCKER_ROOT/$DETECTION_DIR/models \
--volume $PWD/$DETECTION_DIR/code:$DOCKER_ROOT/$DETECTION_DIR/code"

echo "DATA_VOLUME: $DATA_VOLUME"

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

echo "CONTAINER:     $CONTAINER_IMAGE"

# check for V4L2 devices
V4L2_DEVICES=" "

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

echo "V4L2_DEVICES:  $V4L2_DEVICES"

# check for TTY devices to provide Arduino passthrough support
TTY_DEVICES=" "

for i in {0..9}
do
	if [ -a "/dev/ttyUSB$i" ]; then
		TTY_DEVICES="$TTY_DEVICES --device /dev/ttyUSB$i "
	fi
done

for i in {0..9}
do
	if [ -a "/dev/ttyACM$i" ]; then
		TTY_DEVICES="$TTY_DEVICES --device /dev/ttyACM$i "
	fi
done

echo "TTY_DEVICES:  $TTY_DEVICES"

# run the container
sudo xhost +si:localuser:root

echo "Please make sure compose file is using $CONTAINER_IMAGE"

