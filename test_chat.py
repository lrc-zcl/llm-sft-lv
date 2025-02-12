import torch
from trl import get_kbit_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
保存完成merge model 之后进行对话测试
"""

model_path = "/home/lv/llm-demo/merge_models/epochs_1"

tokenizers = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=get_kbit_device_map(),
                                             torch_dtype=torch.float16)

if __name__ == "__main__":
    model = model.eval()
    response, history = model.chat(tokenizers, "小明家里有3个苹果，他给了小红1个，小明还剩几个苹果？", history=[])
    print(response)
