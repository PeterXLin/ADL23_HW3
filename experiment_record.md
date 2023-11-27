- Show your performance:
  - What is the final performance of your model on the public testing set? (2%)
| run name| step | perplexity |
| ------------- | --- | ----- |
| original      | X   | 5.462 |
| finetune 4    | 100 | 3.98  |
| full finetune | 100 | 3.81  |
| small lr(3e-5)| 200 | 3.89  |
| warm up(1e-4) | 200 | 3.679 |
| 2e-4          | 50 | 3.80  |
| finetune final (r8) | 250 | 3.70 |


-> learning rate 2e-5, 100 step is enough
-> set leaning rate 1.4e-4, 200 step


[huggingface/peft/example/fp4_finetuning](https://github.com/huggingface/peft/blob/main/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py)
[how to use qlora](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
