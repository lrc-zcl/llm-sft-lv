import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import get_kbit_device_map


def merge_models(base_model_path, lora_adapter_path, final_save_path):
    """
    和合并微调后的adapter模型文件和基础的base_model
    """
    lora_tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path, use_fast=True,
                                                   trust_remote_code=True)  # 加载adapter 分词器

    base_model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": get_kbit_device_map()
    }
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **base_model_kwargs)

    merge_model = PeftModel.from_pretrained(base_model, lora_adapter_path, device_map=get_kbit_device_map())
    merge_model = merge_model.merge_and_unload() # 合并模型

    merge_model.save_pretrained(final_save_path)
    lora_tokenizer.save_pretrained(final_save_path)

if __name__ == "__main__":
    merge_models(base_model_path="/home/lv/chatglm3-6b",lora_adapter_path="/home/lv/llm-demo/outputs/best_models",final_save_path="/home/lv/llm-demo/merge_models/epochs_1")
    print("merge model successfully!")