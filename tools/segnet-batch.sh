#!/bin/bash  

SEQ_IN=$1

FILES="$SEQ_IN/*.png"

for f in $FILES
do
	echo "Processing $f"
	./segnet-console $f $f $2
done


