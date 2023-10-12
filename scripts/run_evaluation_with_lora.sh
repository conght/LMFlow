#!/bin/bash

# --model_name_or_path specifies the original huggingface model
# --lora_model_path specifies the model difference introduced by finetuning,
#   i.e. the one saved by ./scripts/run_finetune_with_lora.sh


CUDA_VISIBLE_DEVICES=0,1,2 \
    deepspeed examples/evaluation.py \
    --answer_type text \
    --model_name_or_path models/Baichuan-7B \
    --trust_remote_code True \
    --lora_model_path output_models/finetuned_baichuan7b_qlora_20230922_01 \
    --dataset_path data/damage_data/20230915/test \
    --prompt_structure "Input: {input}" \
    --deepspeed examples/ds_config.json \
    --inference_batch_size_per_device 1 \
    --metric neg_log_likelihood
