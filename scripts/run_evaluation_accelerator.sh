#!/bin/bash

if [ ! -d data/MedQA-USMLE ]; then
  cd data && ./download.sh MedQA-USMLE && cd -
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file configs/accelerator_singlegpu_config.yaml examples/evaluation.py \
    --answer_type text \
    --model_name_or_path models/Baichuan2-13B-Base \
    --trust_remote_code True \
    --lora_model_path output_models/finetuned_baichuan2_13b_qlora_20230920_01 \
    --dataset_path data/damage_data/20230915/test \
    --use_ram_optimized_load True \
    --deepspeed examples/ds_config.json \
    --metric accuracy \
    --output_dir output_dir/accelerator_1_card \
    --inference_batch_size_per_device 1 \
    --use_accelerator_for_evaluator True \
    --temperature 0.1 \
    --torch_dtype bfloat16
