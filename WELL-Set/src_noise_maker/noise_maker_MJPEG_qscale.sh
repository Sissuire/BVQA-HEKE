#!/bin/bash

echo "Making noise for MJPEG compression with controlling Q_scale"

mkdir -p ../data/MJPEG_qscale               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/MJPEG_qscale/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 5 10 15 20 25 30              # param
    do
        dst="${dstp}${fileid}_MJPEG_${i}.mjpeg"     # post-fix
        ffmpeg -v quiet -y -i ${src} -c:v mjpeg -qscale ${i} ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
