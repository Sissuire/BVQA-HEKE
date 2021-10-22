#!/bin/bash

echo "Making noise for HEVC compression with controlling QP"

mkdir -p ../data/HEVC_qp               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/HEVC_qp/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 23 28 32 36 39 42      # param
    do
        dst="${dstp}${fileid}_HEVC_qp_${i}.265"
        ffmpeg -v error -y -i ${src} -c:v libx265 -g 72 -x265-params "qp=${i}" -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
