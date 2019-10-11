#!/bin/bash  

SEQ_IN=$1
OUTPUT=$2

FILES="$SEQ_IN/*.png"

for file_in in $FILES
do
	filename=`basename $file_in`
	file_out="$OUTPUT/$filename"
     echo "Processing $filename"
	depthnet-console $file_in $file_out
done


