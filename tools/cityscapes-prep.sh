#!/bin/bash  

h=`pwd`
c=~/Downloads/cityscapes

mkdir train
mkdir train/images
mkdir train/labels
mkdir val
mkdir val/images
mkdir val/labels

for dir in leftImg8bit/train/*/
do
	echo $dir

	cd $dir
	mmv \*_leftImg8bit.png \#1.png
	mv * $c/train/images
	cd $h
done

for dir in leftImg8bit/train_extra/*/
do
	echo $dir

	cd $dir
	mmv \*_leftImg8bit.png \#1.png
	mv * $c/train/images
	cd $h
done

for dir in leftImg8bit/test/*/
do
	echo $dir

	cd $dir
	mmv \*_leftImg8bit.png \#1.png
	cd $h
done

for dir in leftImg8bit/val/*/
do
	echo $dir

	cd $dir
	mmv \*_leftImg8bit.png \#1.png
	mv * $c/val/images
	cd $h
done




for dir in gtCoarse/train/*/
do
	echo $dir

	cd $dir
	rm *instanceIds.png
	rm *labelIds.png
	rm *polygons.json
	mmv \*_gtCoarse_color.png \#1.png
	mv * $c/train/labels
	cd $h
done

for dir in gtCoarse/train_extra/*/
do
	echo $dir

	cd $dir
	rm *instanceIds.png
	rm *labelIds.png
	rm *polygons.json
	mmv \*_gtCoarse_color.png \#1.png
	mv * $c/train/labels
	cd $h
done


for dir in gtCoarse/val/*/
do
	echo $dir

	cd $dir
	rm *instanceIds.png
	rm *labelIds.png
	rm *polygons.json
	mmv \*_gtCoarse_color.png \#1.png
	mv * $c/val/labels
	cd $h
done
