#!/bin/bash
# Please run this script under ${project_id} in project directory of

# Parses arguments
model_name_or_path=meta-llama/Llama-2-13b-hf
dataset_path=/home/paperspace/LMFlow/alpaca/train
output_dir=output_models/finetune
deepspeed_args="--include=localhost:6,7 --master_port=11000"
config_file=configs/baichuan-7b-pt-qlora.json

while [[ $# -ge 1 ]]; do
  key="$1"
  case ${key} in
    -m|--model_name_or_path)
      model_name_or_path="$2"
      shift
      ;;
    -d|--dataset_path)
      dataset_path="$2"
      shift
      ;;
    -o|--output_lora_path)
      output_dir="$2"
      shift
      ;;
    --lora_target_modules)
      lora_target_modules="$2"
      shift
      ;;
    --deepspeed_args)
      deepspeed_args="$2"
      shift
      ;;
    --config)
      config_file="$2"
      shift
      ;;
    *)
      echo "error: unknown option \"${key}\"" 1>&2
      exit 1
  esac
  shift
done

# Finetune
exp_id=finetune_with_qlora_baichuan7B_0915
project_dir=$(cd "$(dirname $0)"/..; pwd)
log_dir=${project_dir}/log/${exp_id}
mkdir -p ${output_dir} ${log_dir}

deepspeed ${deepspeed_args} \
  examples/finetune.py ${config_file} \
    | tee ${log_dir}/train.log \
    2> ${log_dir}/train.err
