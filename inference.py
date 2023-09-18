from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel
import argparse
from get_config import get_cfg
import copy

def parse_args():
    parser = argparse.ArgumentParser(description='QLoRA')
    parser.add_argument('--cfg', type=str, required=True, help='配置py文件')
    return parser.parse_args()

args = parse_args()
args = get_cfg(args)


q_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True,
                                  bnb_4bit_compute_dtype=torch.float32)

    

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

print('loading model')
model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=q_config, device_map="auto", trust_remote_code=True)
print('loading peft model')
finetune_model = PeftModel.from_pretrained(model, args.output_dir)
#print('loading model')
#model = AutoModelForCausalLM.from_pretrained(args.model_path, quantization_config=q_config, device_map="auto", trust_remote_code=True)
gen_kwargs = {"max_length": 128, "num_beams": 1, "do_sample": True, "top_p": 0.5,
                      "temperature": 0.5, "logits_processor": None, "eos_token_id": args.eos_token_id, "pad_token_id": args.pad_token_id}
# gen_kwargs = {"max_length": 128, "num_beams": 1, "do_sample": True, "top_k": 1,
#                       "temperature": 0.8, "logits_processor": None, "eos_token_id": args.eos_token_id, "pad_token_id": args.pad_token_id}
while True:
    print('我:', end='')
    try:
        inputs = tokenizer(input(''), return_tensors='pt')
    except:
        print('输入错误，请重新输入')
        continue
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs, **gen_kwargs)
    # print(pred)
    #print('AI(微调前):', end='')
    #print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))

    pred = finetune_model.generate(**inputs, **gen_kwargs)
    # print(pred)
    print('AI(微调后):', end='')
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
