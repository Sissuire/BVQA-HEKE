# BVQA-HEKE

---------------------
Pretrained encoders, VQA features, and validation codes are available now.
-----------------------

### Pretrained model

We provide the pretrained models: HEKE$_c^4$-r2p1d and HEKE$_c^4$-resnet. 

Download the pretrained models from [GoogleDrive](https://drive.google.com/drive/folders/1NzTdDEafcGcHQrMxzc2qAmwWmL9Pk0Xx?usp=sharing) or [百度云](https://pan.baidu.com/s/1ej1snlAM4n1gdMm0AP7AmA)，提取码：2qxh

Put the pretrained models in the folder `./pretrained`, and you can use the model to verify the performance.

### Performance validation

When the pretrained models are put in the folder `./pretrained`, you can 

- (1) extract features with `demo_extract_features.py`. This file also results in completely blind VQA performance. 

`python3 -u demo_extract_features.py --database LIVE --model r2p1d_HEKE_4 --load ./pretrained/r2p1d_model_HEKE4.pkl --save ./data/r2p1d_HEKE_4`

- (2) finetune the regressor with `demo_finetune_VQA.py` to get the intra-dataset performance.
`python3 -u demo_finetune_VQA.py --base_path PATH_TO_DATASET --database LIVE --load PATH_TO_FEATURE --epoch 300 --lr 1e-4 --batch_size BATCH_SIZE --seed 12318 --f1 --f2 --f3 --f4 --fave --fmax --fstd`

- (3) conduct inter-dataset / cross-database experiment by 'demo_crossDB_VQA.py`. Command is simiar and please check the file.

If you are interested in other pretrained models in our paper (e.g., HEKE$_c^1$-r2p1d, HEKE$_c^2$-r2p1d, and HEKE$_c^1$-resnet), please contact me for sharing.

To be constructed ..
