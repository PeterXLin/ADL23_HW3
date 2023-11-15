# ADL23_HW3

# Task

訓練 Taiwan-LLaMa 做翻譯文言文的任務。使用 QLoRA 訓練。

# Dataset

- Training (train.json): 10000 (3000 筆就可以過 baseline)
- test (public): 250
- test (private): 250

# Evaluation

- Perplexity(ppl.py)

# Workflow

## Branch Name Format

    <contributor>-<type>-<description>

### Type

- `feat`: A new feature or functionality (e.g., implement the training/testing pipeline)
- `fix`: A bug fix (e.g., fix the bugs in model architecture)
- `refact`: A code change that neither adds a new feature nor fixes bugs
- `test`: A code change on testing written code (e.g., unit tests)
- `doc`: A documentation change (e.g., README, gitignore, or experiment records)

## Commit Message

    <type>: <description>

# Setup Environment

```shell

```

# How to train

```shell

```

# Inference

How to use trained model to generate translation.
-> Please run the following command to generate translation.

```shell
bash ./download.sh # download model
bash ./run.sh /path/to/model-folder /path/to/input.josn /path/to/output.json # generate translation
```
