#!/bin/bash

echo "Making noise for MJPEG 2000 compression with controlling Q_scale"

mkdir -p ../data/MJ2K_qscale               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/MJ2K_qscale/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 4 8 12 16 22 28      # param
    do
        dst="${dstp}${fileid}_MJ2K_${i}.avi"
        ffmpeg -v quiet -y -i ${src} -c:v jpeg2000 -qscale ${i} ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
