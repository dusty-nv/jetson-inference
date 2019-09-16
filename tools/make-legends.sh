#!/bin/bash
#
# this script saves the legend images from the
# segmentation models to the provided directory.
#
# usage: ./make-legends.sh <output-dir>
# 
ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

OUTPUT_DIR=$1
mkdir $OUTPUT_DIR

function make_legend()
{
	network=$1
	legend=$2
	
	segnet-console --network=$network --legend=$OUTPUT_DIR/$legend $ROOT/../data/images/peds-001.jpg $OUTPUT_DIR/ignore.jpg
}


make_legend "FCN-ResNet18-Cityscapes-512x256" "segmentation-cityscapes-legend.jpg"
make_legend "FCN-ResNet18-DeepScene-576x320" "segmentation-deepscene-legend.jpg"
make_legend "FCN-ResNet18-MHP-512x320" "segmentation-mhp-legend.jpg"
make_legend "FCN-ResNet18-Pascal-VOC-320x320" "segmentation-voc-legend.jpg"
make_legend "FCN-ResNet18-SUN-RGBD-512x400" "segmentation-sun-legend.jpg"

rm $OUTPUT_DIR/ignore.jpg

