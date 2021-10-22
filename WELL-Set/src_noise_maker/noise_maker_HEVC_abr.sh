#!/bin/bash

echo "Making noise for HEVC compression with controlling abr"

mkdir -p ../data/HEVC_abr                     # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/HEVC_abr/"                  # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 150 300 600 1200 2400 3600                # param
    do
        dst="${dstp}${fileid}_HEVC_abr_${i}k.265"   # dist name
        ffmpeg -loglevel error -v error -y -i ${src} -c:v libx265 -g 72 -b:v ${i}k -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
