#!/bin/bash  

SEQ_IN=$1

FILES="$SEQ_IN/GT/LABELS/Stereo_Left/Omni_B/*.png"

for f in $FILES
do
	echo "Processing $f"
	./seg-img-tool $f $f
done

FILES="$SEQ_IN/GT/LABELS/Stereo_Left/Omni_F/*.png"

for f in $FILES
do
	echo "Processing $f"
	./seg-img-tool $f $f
done

FILES="$SEQ_IN/GT/LABELS/Stereo_Left/Omni_L/*.png"

for f in $FILES
do
	echo "Processing $f"
	./seg-img-tool $f $f
done

FILES="$SEQ_IN/GT/LABELS/Stereo_Left/Omni_R/*.png"

for f in $FILES
do
	echo "Processing $f"
	./seg-img-tool $f $f
done


