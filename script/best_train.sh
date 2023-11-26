# export CUDA_VISIBLE_DEVICES=0 && python ./code/qlora_instruction_tuning.py \
#     --data_path "./data/train.json"

python ./code/sft.py \
    --model_path "./checkpoint/Taiwan-LLM-7B-v2.0-chat" \
    --train_path "./data/train.json" \
    --test_path "./data/public_test.json" \
    --output_dir "./checkpoint/finetune_full_module_small_lr" \
    --learning_rate 2e-4 \
    --seq_length 1024 \
    --num_train_epochs 1 \
    --use_peft True \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 2 \
    --peft_lora_r 4 \
    --log_with "wandb" \
    --run_name "finetune_full_module_small_lr" \
    --do_eval True \
    --evaluation_strategy "steps" \
    --eval_steps 50
    # --max_train_samples 4 \
    # --max_eval_samples 4 \