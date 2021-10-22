#!/bin/bash

echo "Making noise for H.264 compression with controlling QP"

mkdir -p ../data/H264_qp               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/H264_qp/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 23 28 32 36 39 42      # param
    do
        dst="${dstp}${fileid}_H264_qp_${i}.264"
        #echo "ffmpeg -v error -y -i ${src} -c:v libx264 -qp ${i} -f rawvideo ${dst}"
        ffmpeg -v error -y -i ${src} -c:v libx264 -g 54 -qp ${i} -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
