base_model=$1
peft_model=$2
input=$3
output=$4

export CUDA_VISIBLE_DEVICES=1 && python ./inference.py --base_model_name_or_path $base_model --peft_model_path $peft_model --input_path $input --output_path $output
