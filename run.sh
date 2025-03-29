CUDA_VISIBLE_DEVICES=0,1 python main.py --data ./data/med/ \
--epochs_gat 1 --epochs_conv 1 --weight_decay_gat 0.00001 \
--pretrained_emb False \
--get_2hop True --partial_2hop True --batch_size_gat 47296 --margin 1 \
--out_channels 50 --drop_conv 0.3 \
--output_folder ./checkpoints/ | tee result.txt