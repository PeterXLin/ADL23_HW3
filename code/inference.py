from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, AutoTokenizer, set_seed

from trl import SFTTrainer, is_xpu_available, DataCollatorForCompletionOnlyLM
import os, sys

sys.path.append('../ADL23_HW3')
from utils import get_prompt, get_bnb_config


quantization_config = get_bnb_config()
device_map = "auto"
torch_dtype = torch.bfloat16

model_path = os.path.join(os.getcwd(), script_args.model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)   

from peft import PeftModel, PeftConfig

peft_model_id = "./checkpoint/finetune_2/checkpoint-300"
config = PeftConfig.from_pretrained(peft_model_id)


lora_config = LoraConfig(
    r = LORA_R, # the dimension of the low-rank matrices
    lora_alpha = LORA_ALPHA, # scaling factor for the weight matrices
    lora_dropout = LORA_DROPOUT, # dropout probability of the LoRA layers
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["query_key_value"],
)
model = PeftModel.from_pretrained(model, peft_model_id)



