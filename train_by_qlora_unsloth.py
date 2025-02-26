import torch
import json
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from utils_tools.config_args import SelfTrainargs
from utils_tools.dataset import formatting_prompts_func

"""
使用unsloth 工具对Llama-3.2-3B-Instruct 模型进行微调
"""


def formatting_prompts_func_only(data):
    return formatting_prompts_func(data, tokenizer=tokenizer)


if __name__ == "__main__":
    with open("./configs/llama32_3B-instruct_unsloth.json", "r") as f:
        train_dict = json.load(f)
    custom_args = SelfTrainargs(**train_dict)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=custom_args.model_path,
                                                         max_seq_length=custom_args.max_seq_length,
                                                         dtype=torch.float16,
                                                         load_in_4bit=True)
    peft_model = FastLanguageModel.get_peft_model(
        model, r=custom_args.rank, target_modules=custom_args.taret_models,
        lora_alpha=custom_args.lora_alpha,
        lora_dropout=custom_args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=custom_args.use_gradient_checkpointing,
        random_state=custom_args.random_state,
        use_rslora=False,
        loftq_config=None
    )
    print(peft_model.device)
    print(peft_model.device)
    print(peft_model.device)
    dataset = load_dataset(custom_args.datasets_path, split="train")
    dataset = standardize_sharegpt(dataset=dataset)  # 处理数据集
    dataset = dataset.map(formatting_prompts_func_only, batched=True)  # 添加special tokens

    trainer_args = TrainingArguments(
        num_train_epochs=custom_args.num_train_epochs,
        per_device_train_batch_size=custom_args.per_device_train_batch_size,
        gradient_accumulation_steps=custom_args.gradient_accumulation_steps,
        warmup_steps=custom_args.warmup_steps,
        learning_rate=custom_args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=custom_args.logging_steps,
        optim=custom_args.optim,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=custom_args.random_state,
        output_dir=custom_args.output_dir,
        report_to="none")
    train_er = SFTTrainer(model=peft_model,
                          tokenizer=tokenizer,
                          train_dataset=dataset,
                          dataset_text_field="text",
                          max_seq_length=custom_args.max_seq_length,
                          data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
                          dataset_num_proc=2,
                          packing=False,
                          args=trainer_args)

    train_er = train_on_responses_only(trainer=train_er,
                                       instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                                       response_part="<|start_header_id|>assistant<|end_header_id|>\n\n")  # 减少用户输入的数据的loss
    train_result = train_er.train()
    train_er.save_model(trainer_args.output_dir)
    # 保存训练指标
    metrics = train_result.metrics
    train_er.log_metrics("train", metrics)
    train_er.save_metrics("train", metrics)
    train_er.save_state()
    train_er.info("Train successfully!")
