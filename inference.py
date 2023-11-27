import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel, PeftConfig
import torch
from transformers import BitsAndBytesConfig
import pandas as pd
import argparse

# sys.path.append('../ADL23_HW3')
from utils import get_prompt, get_bnb_config

def parse_args():
    parser = argparse.ArgumentParser(description="Use Qlora finetuned model to generate text")
    parser.add_argument("--base_model_name_or_path", type=str, default=None)
    parser.add_argument("--peft_model_path", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    
    # load data
    df = pd.read_json(args.input_path)
    # df = df[:1] # for test

    # load model
    quantization_config = get_bnb_config()
    config = PeftConfig.from_pretrained(args.peft_model_path)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, quantization_config=quantization_config, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    model = model.to("cuda")
    
    # define generation config
    generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=1024,
                temperature=0.9,
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )


    # generate
    outputs_list = []
    for index, row in tqdm(df.iterrows(), total = df.shape[0]):
        inputs = tokenizer(get_prompt(row['instruction']), return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), generation_config=generation_config)
            output = tokenizer.batch_decode(outputs['sequences'].cpu().numpy(), skip_special_tokens=True)[0].removeprefix(get_prompt(row['instruction']))
            output = output.lstrip(" ") # remove leading space
            output = output.replace("<s>", "").replace("</s>", "")
            outputs_list.append(output)

    # save results
    result_df = pd.DataFrame({"id": df['id'], 'output': outputs_list})
    result_df.to_json(args.output_path, orient='records', force_ascii=False)

if __name__ == "__main__":
    main()



