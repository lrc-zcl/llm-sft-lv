import torch
from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments, HfArgumentParser
from typing import Optional


@dataclass
class CustomizedArguments:
    """
    一些自定义参数
    """
    max_seq_length: int = field(metadata={"help": "输入最大长度"})
    dataset_path: str = field(metadata={"help": "训练集。如果task_type=pretrain，请指定文件夹，将扫描其下面的所有jsonl文件"})
    model_path: str = field(metadata={"help": "预训练权重路径"})
    model_name: str = field(metadata={"help": '模型的名称'})
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss"})
    lora_rank: Optional[int] = field(default=64, metadata={"help": "lora rank"})
    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "lora dropout"})
    use_unsloth: Optional[bool] = field(default=False, metadata={"help": "use sloth or not"})
    train_mode: str = field(default="lora", metadata={"help": "训练方式：[qlora, lora]"})


def train_arguments():
    total_parser = HfArgumentParser((CustomizedArguments, TrainingArguments))
    custom_args, train_args = total_parser.parse_json_file(
        json_file='/home/lv/llm-demo/configs/chatglm3-6b_qlora_train_configs.json')
    return custom_args, train_args


class SelfTrainargs:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
