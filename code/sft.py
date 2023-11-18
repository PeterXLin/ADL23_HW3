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


tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_path: Optional[str] = field(default=None, metadata={"help": "the model path"})
    train_path: Optional[str] = field(default=None, metadata={"help": "the path to the training data"})
    train_test_split: Optional[float] = field(default=0.1, metadata={"help": "the train test split ratio"})
    
    do_eval: Optional[bool] = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )

    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})

    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    run_name: Optional[str] = field(default="test", metadata={"help": "the name of the run"})
    
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=32, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )

    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=4, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.log_with == 'wandb':
    os.environ["WANDB_PROJECT"] = "ADL_HW3"

set_seed(script_args.seed)

# Step 1: Load the model
# if script_args.load_in_8bit and script_args.load_in_4bit:
#     raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
# elif script_args.load_in_8bit or script_args.load_in_4bit:
#     quantization_config = BitsAndBytesConfig(
#         load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
#     )
#     # Copy the model to each device
#     device_map = (
#         {"": f"xpu:{Accelerator().local_process_index}"}
#         if is_xpu_available()
#         else {"": Accelerator().local_process_index}
#     )
#     torch_dtype = torch.bfloat16
# else:
#     device_map = None
#     quantization_config = None
#     torch_dtype = None

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

# Step 2: Load the dataset
raw_dataset = load_dataset("json", data_files = {"train": script_args.train_path})
if script_args.train_test_split:
    raw_dataset = raw_dataset['train'].train_test_split(test_size = script_args.train_test_split)
    test_dataset = raw_dataset.pop("test")
    raw_dataset['validation'] = test_dataset

if script_args.max_train_samples is not None:
    raw_dataset['train'] = raw_dataset['train'].select(range(script_args.max_train_samples))

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。USER: {example['instruction'][i]} ASSISTANT: {example['output'][i]}"
        output_texts.append(text)
    return output_texts
response_template = "ASSISTANT:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with, 
    run_name=script_args.run_name,
    save_steps=script_args.save_steps,
    save_total_limit=script_args.save_total_limit,
    push_to_hub=script_args.push_to_hub,
    hub_model_id=script_args.hub_model_id,
    gradient_checkpointing=script_args.gradient_checkpointing,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r, # LoRA 中間層的維度
        lora_alpha=script_args.peft_lora_alpha, # LoRA scaling factor 歸一化參數
        lora_dropout=0.01, # LoRA dropout
        target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"], # 有哪些層要套用 LoRA (attention q, k, output, fc1, fc2)
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=raw_dataset['train'],
    eval_dataset=raw_dataset['validation'] if script_args.do_eval else None,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    peft_config=peft_config,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)