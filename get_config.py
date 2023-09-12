import yaml
from importlib import import_module

def get_cfg(args):
    if args.cfg.endswith('yaml'):
        cfg_dict = get_yaml(args.cfg)
    elif args.cfg.endswith('py'):
        cfg_dict = get_py(args.cfg)
    # print(cfg_dict)
    for key, value in cfg_dict.items():
        args.__setattr__(key, value)
    return args

def get_yaml(filename):
    with open(filename) as f:
        cfg_dict = yaml.load(f.read(), Loader = yaml.FullLoader)
    return cfg_dict

def get_py(filename):
    filename = convert_filename_to_package(filename)
    mod = import_module(filename)
    mod_dict = mod.__dict__
    del_keys = ['__name__', '__doc__', '__package__', '__loader__', '__spec__', '__file__', '__cached__', '__builtins__']
    for del_key in del_keys:
        mod_dict.pop(del_key)
    return mod_dict 


def convert_filename_to_package(filename):
    # 去除文件扩展名
    filename = filename.strip('.py')
    filename = filename.replace('/', '.')
    filename = filename.replace('\\', '.')
    while filename.startswith('.'):
        filename = filename[1:]
    return filename

def get_train_args_dict(args):
    return {
    "output_dir": args.output_dir,
    "per_device_train_batch_size": args.per_device_train_batch_size,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "learning_rate": args.learning_rate,
    "num_train_epochs": args.num_train_epochs,
    "lr_scheduler_type": args.lr_scheduler_type,
    "warmup_ratio": args.warmup_ratio,
    "logging_steps": args.logging_steps,
    "save_strategy": args.save_strategy,
    "save_steps": args.save_steps,
    "optim": args.optim,
    "fp16": args.fp16,
    "remove_unused_columns": args.remove_unused_columns,
    "ddp_find_unused_parameters": args.ddp_find_unused_parameters,
    "seed": args.seed}
