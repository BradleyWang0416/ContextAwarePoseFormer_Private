#!/bin/bash
mode='debug'
# mode='train'

# data_mode="video"
data_mode="image"
if [ "$data_mode" = "video" ]; then
    config="experiments/human36m/human36m_multiframe_byBradley.yaml"
else
    config="experiments/human36m/human36m.yaml"
fi


frame=4
backbone="hrnet_32"
logdir="./logs"

if [ "$mode" = "debug" ]; then
    CUDA_VISIBLE_DEVICES=7 \
    python -m debugpy --listen 5678 --wait-for-client train.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame \
    --data_mode $data_mode \
    --debug
else
    CUDA_VISIBLE_DEVICES=7 \
    python train.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame \
    --data_mode $data_mode
fi
