#!/bin/bash

echo "Making noise for H.264 compression with controlling abr"

mkdir -p ../data/H264_abr                     # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/H264_abr/"                  # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 150 300 600 1200 2400 3600                # param
    do
        dst="${dstp}${fileid}_H264_abr_${i}k.264"   # dist name
        ffmpeg -v error -y -i ${src} -c:v libx264 -g 54 -b:v ${i}k -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
