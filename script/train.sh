# export CUDA_VISIBLE_DEVICES=0 && python ./code/qlora_instruction_tuning.py \
#     --data_path "./data/train.json"

python ./code/sft.py \
    --model_path "./checkpoint/Taiwan-LLM-7B-v2.0-chat" \
    --train_path "./data/train.json" \
    --output_dir "./checkpoint/finetune_1" \
    --seq_length 640 \
    --num_train_epochs 2 \
    --use_peft True \
    --batch_size 4 \
    --peft_lora_r 4 \
    --log_with "wandb" \
    --run_name "test code"