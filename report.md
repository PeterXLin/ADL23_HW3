# Q1: LLM Tuning

- Describe:

  - How much training data did you use? (2%)
  - How did you tune your model? (2%)
  - What hyper-parameters did you use? (2%)

- Show your performance:
  - What is the final performance of your model on the public testing set? (2%)
  - Plot the learning curve on the public testing set (2%)

# Q2: LLM Inference Strategies

- Zero-Shot
  - What is your setting? How did you design your prompt? (1%)
- Few-Shot (In-context Learning)
  - What is your setting? How did you design your prompt? (1%)
  - How many in-context examples are utilized? How you select them? (1%)
- Comparison:
  - What’s the difference between the results of zero-shot, few-shot, and LoRA? (2%)

# Q3: Bonus: Other methods (2%)

- Choose one of the following tasks for implementation
  - Experiments with different PLMs
  - Experiments with different LLM tuning methods
- Describe your experimental settings and compare the results to those obtained from your original methods

# Reference

[huggingface/peft/example/fp4_finetuning](https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py)
