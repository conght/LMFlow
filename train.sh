#./scripts/run_finetune_with_lora.sh   --model_name_or_path models/moss-moon-003-base --dataset_path data/alpaca/train  --output_l    ora_path output_models/finetuned_moss_lora --lora_target_modules qkv_proj
#./scripts/run_finetune_with_qlora.sh   --model_name_or_path models/moss-moon-003-base --dataset_path data/damage_data/20230906/train --output_lora_path output_models/finetuned_moss_lora --lora_target_modules qkv_proj --deepspeed_args "--include=localhost:0,2,3,4,5,6 --master_port=11000"
#./scripts/run_finetune_with_lora.sh   --model_name_or_path models/Baichuan-7B   --dataset_path data/alpaca/train  --output_lora_p    ath output_models/finetuned_baichuan_lora
./scripts/run_finetune_with_qlora.sh --deepspeed_args "--include=localhost:0,1 --master_port=11000" --config
