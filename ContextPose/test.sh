#!/bin/bash
mode='debug'
# mode='test'

data_mode="video"
# data_mode="image"
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
    python -m debugpy --listen 5679 --wait-for-client train.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame \
    --data_mode $data_mode \
    --debug \
    --eval
else
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
    torchrun \
    --nproc_per_node=7 \
    --master_port=2345 \
    train.py \
    --config $config \
    --backbone $backbone \
    --logdir $logdir \
    --frame $frame \
    --data_mode $data_mode \
    --eval
fi
