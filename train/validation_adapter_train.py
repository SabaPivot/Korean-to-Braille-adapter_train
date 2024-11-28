from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from time import time
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import json
import argparse
from ..utils import translate_text, eval, read_korean_braille_pairs

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VEXFZU/t5-xlarge-ko-kb", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="main", required=False, help="Revision name for the checkpoint.")
parser.add_argument("--peft_model", type=str)
parser.add_argument("--validation_data_path", type=str, default="validation_data.json", required=False, help="Path to the validation data.")

args = parser.parse_args()
model_name = args.model_name
peft_model_id = args.peft_model
revision = args.revision

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = PeftModel.from_pretrained(model, peft_model_id)

inputs, targets = read_korean_braille_pairs()
targets = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in targets]
outputs = []
start = time()
for text in tqdm(inputs, desc="Translating"):
    output = translate_text(text, model, tokenizer)
    outputs.append(output)
print(time() - start)

pred = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

correct = eval(pred, targets)

# Bash shell 실행을 위한 script
if correct >= len(targets):
    print("All correct!")
    exit(0)
else:
    print(f"correct:{correct}")
    exit(1)
