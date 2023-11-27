# export CUDA_VISIBLE_DEVICES=0 && python ./code/qlora_instruction_tuning.py \
#     --data_path "./data/train.json"

python ./code/sft.py \
    --model_path "./checkpoint/Taiwan-LLM-7B-v2.0-chat" \
    --train_path "./data/train.json" \
    --test_path "./data/public_test.json" \
    --output_dir "./checkpoint/finetune_taiwan_llama" \
    --learning_rate 1e-4 \
    --warmup_ratio 0.1 \
    --seq_length 640 \
    --max_steps 200 \
    --save_steps 25 \
    --use_peft True \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_batch_size 2 \
    --peft_lora_r 4 \
    --do_eval True \
    --evaluation_strategy "steps" \
    --eval_steps 50
    # --log_with "wandb" \
    # --run_name "finetune_final_1" \
    # --max_train_samples 4 \
    # --max_eval_samples 4 \
