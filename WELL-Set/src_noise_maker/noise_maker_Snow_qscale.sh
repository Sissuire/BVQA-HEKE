#!/bin/bash

echo "Making noise for Snow compression with controlling qscale"

mkdir -p ../data/Snow_qscale               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/Snow_qscale/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 4 7 10 13 18 24              # param
    do
        dst="${dstp}${fileid}_Snow_${i}.avi"     # post-fix
        ffmpeg -v quiet -y -i ${src} -c:v snow -g 64 -q:v ${i} ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
