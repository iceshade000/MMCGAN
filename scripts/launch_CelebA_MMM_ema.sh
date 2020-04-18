#!/bin/bash
CUDA_VISIBLE_DEVICES=1,2 python train.py \
--shuffle --batch_size 128 --parallel \
--num_G_accumulations 1 --num_D_accumulations 1 --num_epochs 200 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4  --D_B2 0.999 --G_B2 0.999 \
--dataset CelebA \
--G_ortho 0.0 \
--G_attn 0 --D_attn 0 \
--G_init N02 --D_init N02 \
--ema --use_ema --ema_start 1000 \
--test_every 2000 --save_every 2000 --num_best_copies 1 --num_save_copies 1 --seed 0 \
--model MMMGAN --thrd 100 --anchor 1 --G_param SN --resume