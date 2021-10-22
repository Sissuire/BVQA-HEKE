All information about WELL-Set (a WEakly Labeled Large-scale dataSet) is given here.

### Summary 

WELL-Set is a synthetic video quality dataset, where the reference videos are mainly collected from websites ([YouTube](www.youtube.com) and [Bilibili](www.bilibili.com)), TV plays, films, and sports. We degrade the reference videos with various compression and transmission artifacts. And the label is given with reference-based VQA methods. Typically, we consider the following four VQA methods for weak labels:
- MS-SSIM
- GMSD
- ST-GMSD
- ST-RRED

Other methods with robust performance and computational efficiency are also good.

### Release

I truly understand the importance of data in this area, but since the reference videos are collected from various websites, we must notice the copyright.

Instead of releasing the data directly, we provide all the sources (including the website links, degradation and labeling codes) to encourage anyone interested to download the videos, generating the data, and computing the weak labels. This procedure would last several days.

#### 1. Download videos 

In the file `urls.txt`, we provide the links for videos from YouTube and Bilibili. 

You can use [youtube-dl](https://github.com/ytdl-org/youtube-dl) to get the videos from YouTube with the best quality
`youtube-dl -g -f bestvideo ${URL_LINK}`

You can use [you-get](https://pypi.org/project/you-get/) to get the videos from Bilibili with the best quality
`you-get ${URL_LINK} -o ${OUTPUT_FILE}`

NOTE: We also include other videos, including dozens of TV plays, Basketball (NBA) and Football matches to enrich the content, but the links are not provided. You can include your source videos which are easy to access.  

#### 2. Preprocessing

For each source video, we use [FFmpeg](https://ffmpeg.org/) to segment, resize, and transcode the video. Consider the memory consumption and the consistency with VQA databases, we commonly resize the video into the size `768x432`, but we recommend you consider a proper frame size for your purpose. 

For simplicity, you can fix the frame rate (25fps), frame size (`768x432`), and the duration (250 frames correspond to 10s). 
`ffmpeg -i ${SOURCE_FILE} -loglevel warning -hide_banner -ss ${SEG_POS} -vframes 250 -r 25 -an -c:v libx264 -crf 8 -s 768x432 ${OUTPUT_FILE}.mp4`

The only parameter for you is the `${SEG_POS}` which indicates the start position of the sementation provided as `hh:mm:ss` (for example, `00:01:30`). For videos with short duration, you can segment one or two video clips, and for a longer duration with variety, you can segment more 10s clips as the reference.

We uniformly transcode the video with `libx264` for a fast implementation.

#### 3. Generating samples

We implement various compression and transmission artifacts with `FFmpeg`. Please check the codes in the folder `./src_noise_maker`. 

Run `noise_maker_all_compression.sh` and `noise_maker_PacketLoss_H264_crf` for the sample generation. In the code, the file `list.txt` contains the reference clips. The code reads each reference clip and generates the corresponding degraded data in particular folders.

#### 4. Weakly labeling

Once you generate the degraded data, it is easy to use FR-VQA models to compute the weak labels. The codes are provided in the folder `./fr_vqa_for_weak_labels`. See the `demo_run.m` in the folder.

Once the FR-VQA labels are computed, we compute statistics of the FR-VQA methods on LIVE-VQA database, and use the mean, min, and max values to rescale the weak labels `L = 0.5 + 8.5 * (L - Lmin) / (Lmax - Lmin)`. Please check the file `dataset.py` (the comment content in [Line 458](https://github.com/Sissuire/BVQA-HEKE/blob/5b5b7f168fbafe651380bbcc21f778e1a722dfa3/dataset.py#L458) and [Line 472](https://github.com/Sissuire/BVQA-HEKE/blob/5b5b7f168fbafe651380bbcc21f778e1a722dfa3/dataset.py#L472)) in the previous directory.
