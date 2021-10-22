#!/bin/bash

echo "Making Gaussian noise for H.264 compression"

mkdir -p ../data/Gaussian               # dst path 

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp="../data/Gaussian/"          # dst path setting
    echo "processing [src]: ${src}, [dst_path]: ${dstp}"

    # run
    for i in 7 10 13 16 21 26      # param
    do
        dst="${dstp}${fileid}_Gaussian_${i}.264"
        #echo "ffmpeg -v error -y -i ${src} -c:v libx264 -qp ${i} -f rawvideo ${dst}"
        ffmpeg -v error -y -i ${src} -c:v libx264 -crf 23 -vf noise=alls=${i}:allf=t -f rawvideo ${dst} < /dev/null  # fix input stream or redict file stream
    done

done < "../data/list.txt"
