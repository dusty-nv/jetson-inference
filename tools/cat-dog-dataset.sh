#!/bin/bash
#
# This script randomly generates the cat/dog dataset from
# a larger subset of ILSVRC12.  To use it, first download
# the ILVSRC12 subset (22.5GB) from this Google Drive link: 
#
# https://drive.google.com/open?id=1LsxHT9HX5gM2wMVqPUfILgrqVlGtqX1o
#
# Then extract this archive and substitue it's location in
# the IMAGENET_DIR variable below.  Then create an empty
# folder for cat_dog, and substitue that in OUTPUT_DIR.
#
# The script will create subdirectories for train, val,
# and test underneath the OUTPUT_DIR, and then fill those
# directories with the specified number of images for each.
#
# These images are pulled randomly from the cat and dog
# directories under IMAGENET_DIR, and you can change the
# size of the dataset by modifying the NUM_TRAIN, NUM_VAL,
# and NUM_TEST variables below.
#
IMAGENET_DIR=~/nvidia/datasets/imagenet/ilsvrc12_subset
OUTPUT_DIR=~/nvidia/datasets/cat_dog

NUM_TRAIN=2500
NUM_VAL=600		# make this NUM_TEST more than you want (500),
NUM_TEST=100		# because the test dataset is moved out of val


# https://unix.stackexchange.com/a/217720
function random_copy()
{
	cd $1
	shuf -zn$3 -e *.$4 | xargs -0 cp -vt $2
}

function random_move()
{
	cd $1
	shuf -zn$3 -e *.$4 | xargs -0 mv -vt $2
}

function rename_extensions()
{
	cd $1
	rename "s/.$2/.$3/" *.$2
}

function rename_sequential()
{
	cd $1
	ls -v | cat -n | while read n f; do mv -n "$f" `printf "%0$2d.jpg" $n`; done 
}

function setup_dirs()
{
	mkdir $OUTPUT_DIR/$1
	mkdir $OUTPUT_DIR/$1/cat
	mkdir $OUTPUT_DIR/$1/dog
}

setup_dirs "train"
setup_dirs "test"
setup_dirs "val" 

random_copy $IMAGENET_DIR/train/cat $OUTPUT_DIR/train/cat $NUM_TRAIN "jpg"
random_copy $IMAGENET_DIR/train/dog $OUTPUT_DIR/train/dog $NUM_TRAIN "jpg"

random_copy $IMAGENET_DIR/val/cat $OUTPUT_DIR/val/cat $NUM_VAL "JPEG"
random_copy $IMAGENET_DIR/val/dog $OUTPUT_DIR/val/dog $NUM_VAL "JPEG"

rename_extensions $OUTPUT_DIR/val/cat "JPEG" "jpg"
rename_extensions $OUTPUT_DIR/val/dog "JPEG" "jpg"

random_move $OUTPUT_DIR/val/cat $OUTPUT_DIR/test/cat $NUM_TEST "jpg"
random_move $OUTPUT_DIR/val/dog $OUTPUT_DIR/test/dog $NUM_TEST "jpg"

rename_sequential $OUTPUT_DIR/train/cat 4
rename_sequential $OUTPUT_DIR/train/dog 4

rename_sequential $OUTPUT_DIR/val/cat 3
rename_sequential $OUTPUT_DIR/val/dog 3

rename_sequential $OUTPUT_DIR/test/cat 2
rename_sequential $OUTPUT_DIR/test/dog 2

