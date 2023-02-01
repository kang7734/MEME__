# MEME:Multi-Encoder Multi-Expert framework with data augmentation for video retrieval
(January 25, 2023) First Version

## X-CLIP, CLIP4Clip has their own License and we applied MEME and GPS on top of the code provided by the baseline models.

The implementation of paper 

This is the PyTorch code of the MEME. The code has been tested on PyTorch 1.7.1.

Our method experiments on MSR-VTT, MSVD, and LSMDC based on three models, [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), [ts2net](https://github.com/yuqi657/ts2_net), and [X-CLIP](https://github.com/xuguohai/X-CLIP), and achieves State-of-the-art with R@1 of 49.0 using ViT-B-32.pt on the MSR-VTT dataset.

## Introduction

## Requirement
for [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
```shell
# From CLIP
cd CLIP4Clip
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```
for [ts2net](https://github.com/yuqi657/ts2_net)
```
# From CLIP
cd ts2net
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install opencv-python boto3 requests pandas
```
for [X-CLIP](https://github.com/xuguohai/X-CLIP)
```
# From X-CLIP
cd X-CLIP
pip install -r requirements.txt
```

## Data Preparing, Usage
Please refer to [ArrowLuo/CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) to get data annotation.

### For MSR-VTT

for [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
```shell
DATA_PATH=[Your MSRVTT data and videos path]
python -m torch.distributed.launch --nproc_per_node=2 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos \
--output_dir ckpts/ckpt_msrvtt_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

for [ts2net](https://github.com/yuqi657/ts2_net)
```shell
sh scripts/run_msrvtt.sh
```

for [X-CLIP](https://github.com/xuguohai/X-CLIP)
```shell
sh scripts/run_xclip_msrvtt_vit32.sh
```

### For MSVD

for [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
```shell
DATA_PATH=[Your MSVD data and videos path]
python -m torch.distributed.launch --nproc_per_node=2 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/MSVD_Videos \
--output_dir ckpts/ckpt_msvd_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msvd \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0 --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```

for [X-CLIP](https://github.com/xuguohai/X-CLIP)
```shell
sh scripts/run_xclip_msvd_vit32.sh
```

### For LSMDC

for [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
```shell
DATA_PATH=[Your LSMDC data and videos path]
python -m torch.distributed.launch --nproc_per_node=2 \
main_task_retrieval.py --do_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/LSMDC_Videos \
--output_dir ckpts/ckpt_lsmdc_retrieval_looseType \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype lsmdc --feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32
```


for [X-CLIP](https://github.com/xuguohai/X-CLIP)
```shell
sh scripts/run_xclip_lsmdc_vit32.sh
```

## Text-Video Retreival Results

| Dataset |  Model  | R@1 | R@5 |  R@10  |  MeanR  |
| :---         |     :---:      |     ---: |     :---:      |          ---: |      ---: |
| MSR-VTT   |  CLIP4Clip(Original)  |  43.1    |   72.7     |      81.5      |     15.7    |
|      |    CLIP4Clip(+MEME) |     **45.0(+1.9)**     |     72.3(-0.4)    |   **82.2(+0.7)**   |      **13.7(-2.0)**     |
|      |    ts2net(+Original) |   46.4     |   74.7      |   82.8      |   14.0      |
|      |    ts2net(+MEME) |    **46.6(+0.2)**     |    73.1(-1.6)    |      **82.9(+0.1)**     |      **12.6(-1.4)**   |
|      |    X-CLIP(+Original) |   46.8     |     73.8       |      83.1     |    13.1   |
|      |    X-CLIP(+MEME) |     **49.0(+2.2)**       |    73.5(-0.3)        |       82.0(-1.1)      |    **13.0(-0.1)**   |
| MSVD   |  CLIP4Clip(Original)  |   45.7   |      75.6    |       84.0       |     10.6        |
|      |    CLIP4Clip(+MEME) |      **45.8(+0.1)**        |     75.4(-0.2)       | **84.2(+0.2)**  | **0.3(-0.3)**         |
|      |    ts2net(+Original) |      45.1      |      75.5     |    84.5       |    10.2       |
|      |    ts2net(+MEME) |      **45.4(+0.3)**       |   **75.9(+0.4)**      |      **84.7(+0.2)**      |       10.2(±0.0)     |
|      |    X-CLIP(+Original) |     46.4     |      76.4      |     84.6      |    9.8  |
|      |    X-CLIP(+MEME) |    **46.6(+0.2)**      |     **76.5(+0.1)**       |      **85.0(+0.4)**       |   10.0(+0.2)    |
| LSMDC   |  CLIP4Clip(Original)  |  23.0    |       40.9    |     48.4         |      58.8       |
|      |    CLIP4Clip(+MEME) |      **23.4(+0.4)**        |   **42.0(+1.1)**         |     **49.3(+0.9)**    |      **57.3(-1.5)**          |
|      |    ts2net(+Original) |      20.4      |   40.1      |     47.5      |    68.3    |
|      |    ts2net(+MEME) |**21.3(+0.9)**| **40.8(+0.7)**  |  47.3(-0.2) |   68.3(±0.0).  |
|      |    X-CLIP(+Original) |   23.2   |41.0|51.1|55.8|
|      |    X-CLIP(+MEME) |**24.3(+1.1)**|**42.2(+1.2)**|**51.7(+0.6)**|**53.8(-2.0)**|


## Video-Text Retreival Results

| Dataset |  Model  | R@1 | R@5 |  R@10  |  MeanR  |
| :---         |     :---:      |     ---: |     :---:      |          ---: |      ---: |
| MSR-VTT   |  CLIP4Clip(Original)  |  43.4    |   70.0     |      80.3      |   11.8    |
|      |    CLIP4Clip(+MEME) |**42.5(-0.9)**|**71.0(+1.0)**|**81.4(+1.1)**|**10.3(-1.5)**|
|      |    ts2net(+Original) |45.6|73.1|83.4|9.6|
|      |    ts2net(+MEME) |**45.8(+0.2)**|71.8(-1.3)|**83.7(+0.3)**|**8.4(-1.2)**|
|      |    X-CLIP(+Original) |47.3|73.6|81.8|9.6|
|      |    X-CLIP(+MEME) |**47.7(+0.4)**|**74.0(+0.4)**|**83.3(+1.5)**|**9.4(-0.2)**|
| MSVD   |  CLIP4Clip(Original)  |49.9|70.8|76.8|15.2|
|      |    CLIP4Clip(+MEME) |**59.1(+9.2)**|**81.7(+10.9)**|**87.8(+11.0)**|**6.8(-8.4)**|
|      |    ts2net(+Original) |56.4|79.1|85.2|9.4|
|      |    ts2net(+MEME) |**58.1(+1.7)**|**84.6(+5.5)**|**89.0(+3.8)**|**6.0(-3.4)**|
|      |    X-CLIP(+Original) |53.9|79.0|85.3|7.1|
|      |    X-CLIP(+MEME) |**63.8(+9.9)**|**87.8(+8.8)**|**92.5(+7.2)**|**4.2(-2.9)**|
| LSMDC   |  CLIP4Clip(Original)  |20.4|39.6|49.3|54.2|
|      |    CLIP4Clip(+MEME) |**21.1(+0.7)**|**40.4(+0.8)**|49.2(-0.1)|**53.1(-1.1)**|
|      |    ts2net(+Original) |20.5|37.3|46.4|62.4|
|      |    ts2net(+MEME) |**20.8(+0.3)**|**37.8(+0.5)**|**47.3(+0.9**)|64.1(+1.7)|
|      |    X-CLIP(+Original) |22.4|40.4|48.7|51.7|
|      |    X-CLIP(+MEME) |22.5(+0.1)|41.7(+1.3)|50.5(+1.8)|49.7(-2.0)|




## Citation

## Acknowledge
Our code is based on [ArrowLuo/CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), [yuqi657/ts2net](https://github.com/yuqi657/ts2_net), [xuguohai/X-CLIP](https://github.com/xuguohai/X-CLIP)
