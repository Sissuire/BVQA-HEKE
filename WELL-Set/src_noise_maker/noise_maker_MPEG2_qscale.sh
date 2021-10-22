#!/bin/bash

echo "Making noise for MPEG-2 compression with controlling Q_scale"

mkdir -p ../data/MPEG2_qscale               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/MPEG2_qscale/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 5 10 15 20 25 29              # param
    do
        dst="${dstp}${fileid}_MPEG2_${i}.m2v"     # post-fix
        ffmpeg -v quiet -y -i ${src} -c:v mpeg2video -g 64 -q:v ${i} ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
