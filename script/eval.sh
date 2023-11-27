export CUDA_VISIBLE_DEVICES=1 && python ./ppl.py \
    --base_model_path "./checkpoint/Taiwan-LLM-7B-v2.0-chat" \
    --peft_path "./adapter_checkpoint" \
    --test_data_path "./data/public_test.json"
