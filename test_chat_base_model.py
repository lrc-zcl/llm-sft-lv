import torch
from trl import get_kbit_device_map
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
使用base model进行对话测试
"""
model_path = "/home/lv/chatglm3-6b"

tokenizers = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=get_kbit_device_map(),
                                             torch_dtype=torch.float16)

if __name__ == "__main__":
    model = model.eval()
    response, history = model.chat(tokenizers, "小明家里有3个苹果，他给了小红1个，小明还剩几个苹果？", history=[])
    print(response)
