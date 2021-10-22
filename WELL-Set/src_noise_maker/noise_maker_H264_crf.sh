#!/bin/bash

echo "Making noise for H.264 compression with controlling crf"

mkdir -p ../data/H264_crf                     # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/H264_crf/"                  # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 23 28 32 36 39 42                # param
    do
        dst="${dstp}${fileid}_H264_crf_${i}.264"   # dist name
        ffmpeg -v error -y -i ${src} -c:v libx264 -g 54 -crf ${i} -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
