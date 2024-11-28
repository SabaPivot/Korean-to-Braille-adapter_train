from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from time import time
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import argparse
import os
from utils import translate_text, eval, read_json

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VEXFZU/t5-xlarge-ko-kb", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="main", required=False, help="Revision name for the checkpoint.")
parser.add_argument("--peft_model_path", default="/success_adapter", type=str)
parser.add_argument("--benchmark_data_path", type=str, default="benchmark.json")

args = parser.parse_args()
model_name = args.model_name
peft_model_path = args.peft_model_path
revision = args.revision
data_path = args.benchmark_data_path

tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

inputs, targets = read_json(data_path)
targets = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in targets]
outputs = []

special_characters = {'<', '>', '《', '》', '『', '』'} 
special_characters_path = os.path.join(peft_model_path, 'brackets/LoRA, LR=5e-5, Epoch=19')

currency_character = {'$', '￥', '€'}
currency_character_path = os.path.join(peft_model_path, 'euro_dollar_yen/LoRA, LR=5e-5, Epoch=30')

start = time()
for text in tqdm(inputs, desc="Translating"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
    if any(char in special_characters for char in text):
        model = PeftModel.from_pretrained(model, special_characters_path)

    elif any(char in currency_character for char in text):
        model = PeftModel.from_pretrained(model, currency_character_path)
    
    else:
        pass
    output = translate_text(text, model, tokenizer)
    outputs.append(output)
print(time() - start)

pred = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

correct = eval(pred, targets)