#!/bin/bash
# This script produces a time series of a set of images or a movie

if [ "$#" -ne 3 ] 
    then
        echo "Usage: $0 <image_sequence> <row number> <output image>"
        exit 1
fi

fname=$1
row=$2
tsname=$3

ffmpeg -i $fname -f image2pipe -vcodec ppm \
  -vf crop=1292:2:3:$row - | montage - -tile 1x -geometry +0+0 $tsname
