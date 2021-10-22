#!/bin/bash

echo "Making noise ..."

bash noise_maker_Gaussian.sh
bash noise_maker_H264_abr.sh
bash noise_maker_H264_crf.sh
bash noise_maker_H264_qp.sh
bash noise_maker_HEVC_abr.sh
bash noise_maker_HEVC_crf.sh
bash noise_maker_HEVC_qp.sh
bash noise_maker_MJ2K_qscale.sh
bash noise_maker_MJPEG_qscale.sh
bash noise_maker_MPEG2_qscale.sh
bash noise_maker_MPEG4_abr.sh
bash noise_maker_MPEG4_qscale.sh
bash noise_maker_Snow_qscale.sh

echo "Done for all."

