#!/bin/bash
mode='debug'
# mode='train'
# mode='train_nohup'

config=experiments/human36m/vqvae_byBradley.yaml


frame=16
backbone="hrnet_32"
logdir="./logs/vqvae_byBradley/${backbone}_f${frame}_250914_17pm"

if [ "$mode" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=7 \
    python -m debugpy --listen 5679 --wait-for-client train_vqvae_byBrad.py \
    --config $config \
    --backbone $backbone \
    --logdir "./logs/tmp" \
    --frame $frame
elif [ "$mode" = "train" ]; then
    CUDA_VISIBLE_DEVICES=7 \
    python train_vqvae_byBrad.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame
else
    CUDA_VISIBLE_DEVICES=4,5,6,7 \
    nohup \
    python -u train_vqvae_byBrad.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame \
    > vqvae_${backbone}_f${frame}.log 2>&1 &
fi
