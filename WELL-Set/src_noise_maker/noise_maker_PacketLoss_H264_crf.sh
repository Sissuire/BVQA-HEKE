#!/bin/bash

echo "Making noise for Packet Loss with  H.264 compression controlling crf"

mkdir -p ../data/Packet_Loss_H264_crf                     # dst path 
mkdir -p ../data/Packet_Loss_tmp

gops=(48 72 96)
slices=(16 18 24)

while IFS= read -r line
do
    # 
    fileid=${line%.*}
    file="${fileid}.mp4"
    src="../data/ref/${file}"

    dstp1="../data/Packet_Loss_tmp/"                            # saving original .264
    dstp2="../data/Packet_Loss_H264_crf/"                       # saving Packet Loss dist video
    echo "processing [src]: ${src}, [dst_path]: ${dstp2}"

    b1=3    # offset for gop
    b2=3    # offset for slice
    b3=3    # offset for crf
    # run
    for i in 24 32                # param +/- 1,2
    do
        r1=$RANDOM
        r2=$RANDOM
        r3=$RANDOM

        ocrf=`expr $r1 % $b3`
        idxg=`expr $r2 % $b1`
        idxs=`expr $r3 % $b2`

        crf=`expr $i + $ocrf`
        gop=${gops[${idxg}]}
        slice=${slices[${idxs}]}

        dst1="${dstp1}${fileid}_H264_crf_${crf}.264"   # dist name for original .264
        ffmpeg -v error -y -i ${src} -c:v libx264 -g ${gop} -crf ${crf} -x264opts slice_mode="fixed" -x264opts slices=${slice} -f rawvideo ${dst1} < /dev/null  # fix input stream or redict file stream


        for err in 0.4 1 3 5 10 20
        do
            fff="errp_plr_${err}"
            dst2="${dstp2}${fileid}_PL_H264_crf_${crf}_PL_${err}.264"     # dist name 
             
            r4=$RANDOM
            offset=`expr $r4 % 3900`

            ./pl_simulator $dst1 $dst2 $fff 1 $offset 1 
        done

    done

done < "../data/list.txt"
