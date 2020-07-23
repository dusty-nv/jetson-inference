#!/bin/bash  

SEQ_IN=$1
OUTPUT=$2
NETWORK=$3

FILES="$SEQ_IN/*.jpg"

mkdir $OUTPUT

for file_in in $FILES
do
	filename=`basename $file_in`
	file_out="$OUTPUT/$filename"
     echo "Processing $filename"
	depthnet-console --network=$NETWORK $file_in $file_out
done


