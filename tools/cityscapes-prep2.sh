#!/bin/bash  

h=`pwd`
c=~/Downloads/cityscapes





for dir in gtCoarse/val/*/
do
	echo $dir

	cd $dir
	mv * $c/val/labels
	cd $h
done
