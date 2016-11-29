#!/bin/bash  


SEQ_NUM=$1
SEQ_IN=$2
SEQ_OUT=$3

sudo apt-get update
sudo apt-get install mmv

cd $SEQ_IN/GT/LABELS/Stereo_Left/Omni_B/
mmv 0\* $1SB\#1

cd $SEQ_IN/GT/LABELS/Stereo_Left/Omni_F/
mmv 0\* $1SF\#1

cd $SEQ_IN/GT/LABELS/Stereo_Left/Omni_L/
mmv 0\* $1SL\#1

cd $SEQ_IN/GT/LABELS/Stereo_Left/Omni_R/
mmv 0\* $1SR\#1

cd $SEQ_IN/RGB/Stereo_Left/Omni_B/
mmv 0\* $1SB\#1

cd $SEQ_IN/RGB/Stereo_Left/Omni_F/
mmv 0\* $1SF\#1

cd $SEQ_IN/RGB/Stereo_Left/Omni_L/
mmv 0\* $1SL\#1

cd $SEQ_IN/RGB/Stereo_Left/Omni_R/
mmv 0\* $1SR\#1


mv $SEQ_IN/GT/LABELS/Stereo_Left/Omni_B/* $SEQ_OUT/GT
mv $SEQ_IN/GT/LABELS/Stereo_Left/Omni_F/* $SEQ_OUT/GT
mv $SEQ_IN/GT/LABELS/Stereo_Left/Omni_L/* $SEQ_OUT/GT
mv $SEQ_IN/GT/LABELS/Stereo_Left/Omni_R/* $SEQ_OUT/GT

mv $SEQ_IN/RGB/Stereo_Left/Omni_B/* $SEQ_OUT/RGB
mv $SEQ_IN/RGB/Stereo_Left/Omni_F/* $SEQ_OUT/RGB
mv $SEQ_IN/RGB/Stereo_Left/Omni_L/* $SEQ_OUT/RGB
mv $SEQ_IN/RGB/Stereo_Left/Omni_R/* $SEQ_OUT/RGB
