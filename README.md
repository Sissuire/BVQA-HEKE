# BVQA-HEKE

Source codes for "No-Reference Video Quality Assessment with Heterogeneous Knowledge Ensemble" accepted by ACM-MM 2021, and the extension of this work "Spatiotemporal Representation Learning for Blind Video Quality Assessment" is published in IEEE-TCSVT.

---------------------
Pretrained encoders, VQA features, and validation codes are available now.
-----------------------

### Pretrained model

We provide the pretrained models: HEKE$_c^4$-r2p1d and HEKE$_c^4$-resnet. 

Download the pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1NzTdDEafcGcHQrMxzc2qAmwWmL9Pk0Xx?usp=sharing) or [百度云](https://pan.baidu.com/s/1ej1snlAM4n1gdMm0AP7AmA), 提取码：2qxh

Put the pretrained models in the folder `./pretrained`, and you can use the model to verify the performance.

If you are interested in other pretrained models in our paper (e.g., HEKE$_c^1$-r2p1d, HEKE$_c^2$-r2p1d, and HEKE$_c^1$-resnet), please contact me for sharing.

### PreExtracted features

The VQA features of six synthetic datasets (i.e., LIVE, CSIQ, IVPL, IVC-IC, EPFL-PoliMI, and LIVE-Mobile) are provided. Download the preextracted features from [GoogleDrive](https://drive.google.com/drive/folders/1XArB2E2qd4P0OLBKP0qDck5Ioq66nPAH?usp=sharing) or [百度云](https://pan.baidu.com/s/1OsgYJs5Pi7WZfxhk6M_qrg), 提取码：932a

Currently we only upload the features from r2p1d_HEKE_4. If you need other features, please feel free to contact me.

### Performance validation

When the pretrained models are put in the folder `./pretrained`, you can 

- (1) extract features with `demo_extract_features.py`. This file also results in completely blind VQA performance. 
`python3 -u demo_extract_features.py --database LIVE --model r2p1d_HEKE_4 --load ./pretrained/r2p1d_model_HEKE4.pkl --save ./data/r2p1d_HEKE_4`

- (2) finetune the regressor with `demo_finetune_VQA.py` to get the intra-dataset performance.
`python3 -u demo_finetune_VQA.py --base_path PATH_TO_DATASET --database LIVE --load PATH_TO_FEATURE --epoch 300 --lr 1e-4 --batch_size BATCH_SIZE --seed 12318 --f1 --f2 --f3 --f4 --fave --fmax --fstd`

- (3) conduct inter-dataset / cross-database experiment by 'demo_crossDB_VQA.py`. Command is simiar and please check the file.

If you are interested in other pretrained models in our paper (e.g., HEKE$_c^1$-r2p1d, HEKE$_c^2$-r2p1d, and HEKE$_c^1$-resnet), please contact me for sharing.

### Cite
If you are interested in the work, or find the code helpful, please cite our work
```
@ARTICLE{heke-csvt,
  author={Liu, Yongxu and Wu, Jinjian and Li, Leida and Dong, Weisheng and Zhang, Jinpeng and Shi, Guangming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Spatiotemporal Representation Learning for Blind Video Quality Assessment}, 
  year={2021},
  volume={},
  number={},
  pages={},
  doi={10.1109/TCSVT.2021.3114509}}
```

### Contact

If any question or bug, feel free to contact me via `yongxu.liu@stu.xidian.edu.cn`.
