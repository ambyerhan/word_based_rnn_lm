#!/bin/sh
GPU_ID=0
CEDIM=15
DROP=0.5
LRATE=1.0
LR_DECAY=0.5
DECAY_WHEN=1.0
CLIP=5.0
PARAM_INIT=0.05
LEN=35
BATCH=20
MEPOCH=25
SAVE=True


CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
	--char_embed_size=$CEDIM \
	--dropout=$DROP \
	--learning_rate=$LRATE \
	--learning_rate_decay=$LR_DECAY \
	--decay_when=$DECAY_WHEN \
	--max_grad_norm=$CLIP \
	--param_init=$PARAM_INIT \
	--num_unroll_steps=$LEN \
	--batch_size=$BATCH \
	--max_epochs=$MEPOCH \
	--save=$SAVE
	
