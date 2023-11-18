# ADL23_HW3
#### Task
訓練 Taiwan-LLaMa 做翻譯文言文的任務。使用 QLoRA 訓練。

#### Dataset
- Training (train.json): 10000 (3000 筆就可以過 baseline)
- test (public): 250
- test (private): 250

#### Evaluation
- Perplexity(ppl.py)

---

## Workflow

### Branch Name Format

    <contributor>-<type>-<description>

#### Type

- `feat`: A new feature or functionality (e.g., implement the training/testing pipeline)
- `fix`: A bug fix (e.g., fix the bugs in model architecture)
- `refact`: A code change that neither adds a new feature nor fixes bugs
- `test`: A code change on testing written code (e.g., unit tests)
- `doc`: A documentation change (e.g., README, gitignore, or experiment records)

### Commit Message

    <type>: <description>

## Installation

```shell
# use conda to manage environment
conda create --name adl23 python=3.9
# install packages
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.34.1
pip install bitsandbytes=0.41.1
pip install peft==0.6.0
pip install datasets
pip install accelerate
conda install scipy

conda env export > environment.yaml
```

---

## Quick start

### train

```shell

```

### Inference
How to use trained model to generate translation.
-> Please run the following command to generate translation.

```shell
bash ./download.sh # download model
bash ./run.sh /path/to/model-folder /path/to/input.josn /path/to/output.json # generate translation
```

### Evaluation



# FAQ
When should one opt for the Supervised Fine Tuning Trainer (SFTTrainer) instead of the regular Transformers Trainer when it comes to instruction fine-tuning for Language Models (LLMs)? 
[Answer](https://datascience.stackexchange.com/questions/122164/lmm-fine-tuning-supervised-fine-tuning-trainer-sfttrainer-vs-transformers-tr)

# Reference 
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
https://huggingface.co/docs/trl/sft_trainer#format-your-input-prompts
https://www.datacamp.com/tutorial/fine-tuning-llama-2