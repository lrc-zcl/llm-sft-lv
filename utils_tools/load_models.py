import os
import torch
from transformers import trainer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import get_kbit_device_map
from loguru import logger


class LoadModels():
    def __init__(self, custom_args, train_args):
        self.custom_args = custom_args
        self.train_args = train_args
        self.model_path = self.custom_args.model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_init_model()

    def load_init_model(self):
        torch_dtype = torch.float16 if self.train_args.fp16 else torch.bfloat16
        use_cache = False if self.train_args.gradient_checkpointing else True
        if self.custom_args.train_mode == "lora":
            quantization_config = None
            # model_kwargs = dict(trust_remote_code=True,
            #                     torch_dtype=torch_dtype,
            #                     use_cache=use_cache,
            #                     device_map="cuda:0"
            #                     )
        else:  # qlora
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if self.train_args.fp16 else torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # 启用双重量化技术 4bit--再量化
                bnb_4bit_quant_type="nf4",  # nf4是一种在量化过程中采用非均匀量化方式的技术
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        model_kwargs = dict(trust_remote_code=True,
                            torch_dtype=torch_dtype,
                            use_cache=use_cache,
                            device_map=get_kbit_device_map(),
                            quantization_config=quantization_config
                            )

        self.init_model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        logger.info(f"model loadding in device {self.init_model.device}")

        if self.custom_args.train_mode == "lora":
            self.make_inputs_require_grad()  # LORA训练策略需要使得embedding层可训练
        else:
            self.init_model = prepare_model_for_kbit_training(self.init_model,
                                                              use_gradient_checkpointing=self.train_args.gradient_checkpointing)
        return self.init_model

    def make_inputs_require_grad(self):
        """
        为模型的输入嵌入层启用梯度计算，从而允许在微调过程中更新输入嵌入层的权重。
        :return:
        """
        if hasattr(self.init_model, "enable_input_require_grads"):
            self.init_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            self.init_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    def find_all_linears(self):
        """
        找到模型中的全部全连接层
        :return: list
        """
        assert self.train_args.bf16 or self.train_args.fp16, "模型数据类型须是bf16或者fp16"
        lora_module_names = set()
        for name, module in self.init_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                name = name.split('.')
                lora_module_names.add(name[0] if len(name) == 1 else name[-1])  # 将所有的全连接层名称保存下来

        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        logger.info(f"lora_module_names is {lora_module_names}")
        return list(lora_module_names)

    def load_peft_model(self):
        peft_configs = LoraConfig(
            r=self.custom_args.lora_rank,
            lora_alpha=self.custom_args.lora_alpha,
            target_modules=self.find_all_linears(),
            lora_dropout=self.custom_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(self.init_model, peft_configs)
        logger.info(f"Lora微调的模型参数量是{sum(p.numel() for p in peft_model.parameters())}")
        logger.info(f"Lora微调的模型可训练参数量是{sum(p.numel() for p in peft_model.parameters() if p.requires_grad)}")
        return {
            "peft_model": peft_model,
            "ref_model": None,
            "peft_config": peft_configs
        }
