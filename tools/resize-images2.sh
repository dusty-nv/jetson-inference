#!/bin/bash  

./resize-images.sh train/images 50
./resize-images.sh train/labels 50
./resize-images.sh val/images 50
./resize-images.sh val/labels 50



