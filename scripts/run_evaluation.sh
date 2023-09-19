#!/bin/bash

CUDA_VISIBLE_DEVICES=0 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path models/Baichuan-7B \
    --dataset_path data/damage_data/20230915/test \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric neg_log_likelihood
