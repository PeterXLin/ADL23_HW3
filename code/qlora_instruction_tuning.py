import os
from datasets import load_dataset
from utils import get_bnb_config
from utils import get_prompt
from peft import prepare_model_for_kbit_training
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model


@dataclass
class ModelArguments:
    model_path: Optional[str] = field(
        default="Taiwan-LLM-7B-v2.0-chat", metadata={"help": "Path to model"}
    )


@dataclass
class DataArguments:
    """Arguments related to data"""
    train_path: Optional[str] = field(
        default=None, metadata={"help": "Path to training data"}
    )
    train_test_split: Optional[float] = field(
        default=None, metadata={"help": "Train test split ratio"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """"""
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})


def main():
    """train Taiwan Llama 7B model to translate from Classical Chinese to Chinese"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)


    # load dataset
    raw_dataset = load_dataset("json", data_files = {"train": data_args.train_path})
    if data_args.train_test_split:
        raw_dataset = raw_dataset['train'].train_test_split(test_size = data_args.train_test_split)
        test_dataset = raw_dataset.pop("test")
        raw_dataset['validation'] = test_dataset

    # load tokenizer
    model_path = os.path.join(os.getcwd(), model_args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)


    # preprocess data
    def preprocess_dataset(examples):
        """"""
        return {"input": [get_prompt(example['instruction']) for example in examples]}


    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config = get_bnb_config(),
        torch_dtype = torch.float16
    )

    # Finally, we need to apply some post-processing on the 8-bit model to enable training, 
    # let's freeze all our layers, and cast the layer-norm in `float32` for stability. 
    # We also cast the output of the last layer in `float32` for the same reasons.
    # 為了穩定性，要把一些參數轉成 fp32
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            # 一些計算量小的參數，例如 layernorm，要轉成 fp32
            param.data = param.data.to(torch.float32)

    # output layer 也要轉成 fp32
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)


    # apply LoRA
    config = LoraConfig(
        r=64, # LoRA 中間層的維度
        lora_alpha=32, # LoRA scaling factor 歸一化參數
        lora_dropout=0.01, # LoRA dropout
        target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"], # 有哪些層要套用 LoRA (attention q, k, output, fc1, fc2)
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters()


    # Verifying the datatypes. ==TODO==: remove this later
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    # train
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

def get_accelerate_model(args, checkpoint_dir):
    pass


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


if __name__ == "__main__":
    main() 