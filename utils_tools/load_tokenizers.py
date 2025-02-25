from transformers import AutoTokenizer
from utils_tools.template import template_dict


def load_chatglm3_tokenizer(path):
    """
    加载分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tokenizer


if __name__ == "__main__":
    tokenizer = load_chatglm3_tokenizer(path='/home/lv/chatglm3-6b')
    prefix = tokenizer.get_prefix_tokens()
    system_string = tokenizer.get_command("<|system|>")
    template = template_dict.get('chatglm3')
    init_system_data = tokenizer.encode(template.system)
    inputs_id = prefix + [system_string] + init_system_data
