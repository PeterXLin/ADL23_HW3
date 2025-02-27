# ADL23_HW3
#### Task
訓練 Taiwan-LLaMa 做翻譯文言文的任務。使用 QLoRA 訓練。

#### Dataset
- Training (train.json): 10000 (3000 筆就可以過 baseline)
- test (public): 250
- test (private): 250

## Installation

### Build environment
```shell
# use conda to manage environment
conda create --name adl23 python=3.9
# install packages
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers==4.34.1
pip install bitsandbytes=0.41.1
pip install peft==0.6.0
pip install trl
pip install datasets
pip install accelerate
pip install tqdm
pip install gdown
conda install scipy
```

or

```shell
conda env create -f ./environment.yaml
```



---

## Quick start
run the following command in root folder to train and inference
### train
```shell
bash ./script/train.sh
```

### Inference
```shell
bash ./download.sh # download model
bash ./run.sh /path/to/Taiwan-LLaMa-folder /path/to/adapter_checkpoint /path/to/input.json /path/to/output.json # generate translation
# for example
bash ./run.sh ./checkpoint/Taiwan-LLM-7B-v2.0-chat ./adapter_checkpoint ./data/private_test.json ./prediction.json
```

### Evaluation
```shell
python3 ppl.py \
    --base_model_path /path/to/Taiwan-Llama \
    --peft_path /path/to/adapter_checkpoint/under/your/folder \
    --test_data_path /path/to/input/data
```


# FAQ
When should one opt for the Supervised Fine Tuning Trainer (SFTTrainer) instead of the regular Transformers Trainer when it comes to instruction fine-tuning for Language Models (LLMs)? 
[Answer](https://datascience.stackexchange.com/questions/122164/lmm-fine-tuning-supervised-fine-tuning-trainer-sfttrainer-vs-transformers-tr)

# Reference 
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
https://huggingface.co/docs/trl/sft_trainer#format-your-input-prompts
https://www.datacamp.com/tutorial/fine-tuning-llama-2
