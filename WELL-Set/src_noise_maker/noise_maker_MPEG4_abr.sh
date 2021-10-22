#!/bin/bash

echo "Making noise for MPEG-4 Part-2 compression with controlling abr"

mkdir -p ../data/MPEG4_abr               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/MPEG4_abr/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 240 480 720 1300 2500 4000              # param
    do
        dst="${dstp}${fileid}_MPEG4_abr_${i}k.mp4"     # post-fix
        ffmpeg -v quiet -y -i ${src} -c:v mpeg4 -g 64 -b:v ${i}k ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
