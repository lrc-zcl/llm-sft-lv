import torch
import json
from utils_tools.template import template_dict
from torch.utils.data import Dataset
from utils_tools.load_tokenizers import load_chatglm3_tokenizer
from torch.utils.data import DataLoader
from loguru import logger

"""
chatglm没有固定的chat_template,需要自己自定义，具体可以根据官网的格式进行定义。
具体可以参考如下：
<|system|>
You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.
<|user|>
Hello
<|assistant|>
Hello, I'm ChatGLM3. What can I assist you today?

注意：
例如：qwen2.5这种模型就用固定的chat_template,可以直接使用tokenizer.apply_chat_template()函数进行查看
但是真实在训练qwen2.5的时候建议使用和上述chatglm3相同构建数据集的方法，因为需要inputs_id、attention_mask、target_mask,需要知道数据长度
不过也可以参考unsloth中的方法，直接使用apply_chat_template方法构建，但是此时的数据集单条是一个list
"""

class SchoolMathDataset(Dataset):
    """
    自定义数据集加载方式
    """

    def __init__(self, data_path, model_path, model_name, max_seq_length):
        super(SchoolMathDataset, self).__init__()
        self.path = data_path
        self.model_name = model_name
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.datasets = self.read_json_file()[:40000]
        self.tokenizer = load_chatglm3_tokenizer(self.model_path)
        self.template = template_dict.get(self.model_name)

    def read_json_file(self):
        datasets = []
        with open(self.path, "r", encoding="utf-8") as json_file:
            for lines_data in json_file:
                datasets.append(json.loads(lines_data.strip()))
        logger.info(f"all datasets size is {len(datasets)}")
        return datasets

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        """
        对单个数据进行格式化操作
        :param index:
        :return:
        prefix + <|system|> + system_data_id
        """
        prefix_id = self.tokenizer.get_prefix_tokens()
        system_string_id = [self.tokenizer.get_command("<|system|>")]
        init_system_data = self.tokenizer.encode(self.template.system)

        # 将每行（每轮对话）格式化
        user_input = dict(self.datasets[index])['instruction']  # 系统输入的问题
        assisant_output = dict(self.datasets[index])['output']  # 语音助手的回答
        user_input_id = self.tokenizer.encode(user_input, add_special_tokens=False)
        assistant_id = self.tokenizer.encode(assisant_output, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        final_inputs_id = prefix_id + system_string_id + init_system_data + user_input_id + assistant_id
        target_mask_id = [0] * (len(final_inputs_id) - len(assistant_id)) + [1] * len(assistant_id)

        # 对长度进行截取,避免长度超过最长限制
        final_inputs_id = final_inputs_id[:self.max_seq_length]
        target_mask_id = target_mask_id[:self.max_seq_length]
        attention_mask_id = [1] * len(final_inputs_id)
        assert len(final_inputs_id) == len(target_mask_id) == len(attention_mask_id), "返回的结果必须长度相同"

        dataset_result = {
            "input_ids": final_inputs_id,
            "target_mask": target_mask_id,
            "attention_mask": attention_mask_id
        }
        assert "input_ids" in dataset_result
        assert "target_mask" in dataset_result
        assert "attention_mask" in dataset_result

        # return {
        #     "input_ids": final_inputs_id,
        #     "target_mask": target_mask_id,
        #     "attention_mask": attention_mask_id
        # }
        return dataset_result


class BatchDataCollator():
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, batch):
        batch_inputs_id = []
        batch_attention_mask = []
        batch_target_mask = []

        batch_max_length = max([len(x['input_ids']) for x in batch])
        for signal in batch:
            signal_inputs_id = signal['input_ids']
            signal_target_mask = signal['target_mask']
            signal_attention_mask = signal['attention_mask']
            if not signal_inputs_id:  # 当前一个数据样本是空
                continue
            require_pad_length = self.max_seq_length - len(signal_inputs_id)

            inputs_id_have_pad = signal_inputs_id + [self.pad_token_id] * require_pad_length
            attention_mask_have_pad = signal_attention_mask + [0] * require_pad_length
            target_mask_have_pad = signal_target_mask + [0] * require_pad_length

            batch_inputs_id.append(inputs_id_have_pad)
            batch_attention_mask.append(attention_mask_have_pad)
            batch_target_mask.append(target_mask_have_pad)

        # 将batch 数据转成 tensor
        batch_input_ids_tensor = torch.tensor(batch_inputs_id, dtype=torch.long).to(self.device)
        batch_attention_mask_tensor = torch.tensor(batch_attention_mask, dtype=torch.long).to(self.device)
        batch_target_mask_tensor = torch.tensor(batch_target_mask, dtype=torch.long).to(self.device)
        # 利用target 的数据制作labels
        batch_labels = torch.where(batch_target_mask_tensor == 1, batch_input_ids_tensor, -100)
        return {
            "input_ids": batch_input_ids_tensor,
            "attention_mask": batch_attention_mask_tensor,
            "labels": batch_labels
        }


def formatting_prompts_func(examples, tokenizer):
    """
    格式化输入的样本数据,即添加special tokens
    可以看到在做模型训练时，Llama模型是加了special tokens的
    """
    convos = examples["conversations"]
    text = [tokenizer.apply_chat_template(convo, tokenize=False) for convo in convos]
    return {"text": text}


if __name__ == "__main__":
    def read_json_file(path):
        datasets = []
        with open(path, "r", encoding="utf-8") as json_file:
            for lines_data in json_file:
                datasets.append(json.loads(lines_data.strip()))
        return datasets


    b = read_json_file(path='/home/lv/llm-demo/data/school_math_0.25M.json')
    a = dict(b[1])['instruction']
    pass
