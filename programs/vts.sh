#!/bin/bash
# This script produces a time series of a set of images or a movie

if [ "$#" -ne 3 ] 
    then
        echo "Usage: $0 <image_sequence> <column number> <output image>"
        exit 1
fi

fname=$1
col=$2
tsname=$3

ffmpeg -i $fname -f image2pipe -vcodec ppm \
  -vf crop=1:964:$col:0 - | montage - -tile x1 -geometry +0+0 $tsname
