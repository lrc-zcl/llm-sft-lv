import os
import torch
from transformers import Trainer
from utils_tools.config_args import train_arguments
from utils_tools.dataset import SchoolMathDataset, BatchDataCollator
from utils_tools.load_models import LoadModels
from utils_tools.load_tokenizers import load_chatglm3_tokenizer
from loguru import logger
if __name__ == "__main__":
    custom_args, train_args = train_arguments()
    tokenizer = load_chatglm3_tokenizer(custom_args.model_path)

    peft_model = LoadModels(custom_args, train_args).load_peft_model()['peft_model']




    dataset = SchoolMathDataset(data_path=custom_args.dataset_path, model_path=custom_args.model_path,
                                model_name=custom_args.model_name,
                                max_seq_length=custom_args.max_seq_length)
    batch_data_collator = BatchDataCollator(tokenizer=tokenizer, max_seq_length=custom_args.max_seq_length)
    trainer = Trainer(model=peft_model, train_dataset=dataset, tokenizer=tokenizer, data_collator=batch_data_collator,args=train_args)
    logger.info(f"model train mode is {custom_args.train_mode}")
    trian_result = trainer.train()
    trainer.save_model('/home/lv/llm-demo/outputs/best_models')
    # 保存训练指标
    metrics = trian_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("train_sess")

    # custom_args, train_args = train_arguments()
    # tokenizer = load_chatglm3_tokenizer(custom_args.model_path)
    # peft_model = LoadModels(custom_args, train_args).load_peft_model()['peft_model']
    # dataset = SchoolMathDataset(data_path=custom_args.dataset_path, model_path=custom_args.model_path,model_name=custom_args.model_name,
    #                             max_seq_length=custom_args.max_seq_length)
    # batch_data_collator = BatchDataCollator(tokenizer=tokenizer,max_seq_length=custom_args.max_seq_length)
    # trainer = Trainer(
    #     model=peft_model,
    #     train_dataset=dataset,
    #     data_collator=batch_data_collator,
    #     tokenizer=tokenizer,
    # )
    # trian_result = trainer.train()
    # trainer.save_model('/home/lv/llm-demo/outputs/best_models')
    # # 保存训练指标
    # metrics = trian_result.metrics
    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()
    # print("train_sess")
