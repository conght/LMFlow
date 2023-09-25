#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerator_singlegpu_config.yaml examples/evaluation.py \
    --answer_type text \
    --model_name_or_path models/Baichuan-7B \
    --trust_remote_code True \
    --dataset_path data/damage_data/20230915/test \
    --use_ram_optimized_load True \
    --deepspeed examples/ds_config.json \
    --metric neg_log_likelihood \
    --output_dir output_dir/accelerator_1_card \
    --inference_batch_size_per_device 1 \
    --use_accelerator_for_evaluator True \
    --temperature 0.1 \
    --torch_dtype bfloat16
