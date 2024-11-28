from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load
from time import time
from tqdm import tqdm
from peft import PeftModel, PeftConfig
import json
import argparse
from utils import translate_text, eval

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VEXFZU/t5-xlarge-ko-kb", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="main", required=False, help="Revision name for the checkpoint.")
parser.add_argument("--peft_model", type=str)

args = parser.parse_args()
model_name = args.model_name
peft_model_id = args.peft_model
revision = args.revision

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = PeftModel.from_pretrained(model, peft_model_id)

inputs, targets = (
    ["<도시와 그 불확실한 벽>이 기나긴 추석 연휴 기간 교보문고 베스트셀러 일 위에 올랐다.",
    "한국의 웹 드라마 《고래 먼지》다.",
    "『신경숙의 엄마를 부탁해』 중에서"], 
    ["⠐⠶⠊⠥⠠⠕⠧⠀⠈⠪⠀⠘⠯⠚⠧⠁⠠⠕⠂⠚⠒⠀⠘⠱⠁⠶⠂⠕⠀⠈⠕⠉⠈⠟⠀⠰⠍⠠⠹⠀⠡⠚⠩⠀⠈⠕⠫⠒⠀⠈⠬⠘⠥⠑⠛⠈⠥⠀⠘⠝⠠⠪⠓⠪⠠⠝⠂⠐⠎⠀⠕⠂⠀⠍⠗⠝⠀⠥⠂⠐⠣⠌⠊⠲",
    "⠚⠒⠈⠍⠁⠺⠀⠏⠗⠃⠀⠊⠪⠐⠣⠑⠀⠰⠶⠈⠥⠐⠗⠀⠑⠾⠨⠕⠶⠆⠊⠲",
    "⠰⠦⠠⠟⠈⠻⠠⠍⠁⠺⠀⠎⠢⠑⠐⠮⠀⠘⠍⠓⠁⠚⠗⠴⠆⠀⠨⠍⠶⠝⠠⠎"]
)
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
