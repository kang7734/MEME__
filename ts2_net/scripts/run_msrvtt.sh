CUDA_VISIBLE_DEVICES=0,1
DATA_PATH=[your data path]
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29668 \
main_task_retrieval.py --do_train --eval_in_train --num_thread_reader=8 \
--epochs=5 --batch_size=128 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/compressed_videos \
--output_dir ckpts/msrvtt \
--cross_num_hidden_layers 4 \
--lr 1e-4 --max_words 32 --max_frames 12 --batch_size_val 64 \
--datatype msrvtt --expand_msrvtt_sentences \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqTransf \
--pretrained_clip_name ViT-B/32