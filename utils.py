from evaluate import load
import json
import os
from peft import PeftModel
from transformers import T5ForConditionalGeneration

def read_json(f_path):
  with open(f_path, 'r', encoding='utf-8') as file:
      data = json.load(file)

  inputs = [item['question_text'] for item in data]
  targets = [item['answer'] for item in data]
  
  # 비어 있지 않은 데이터만 읽어오기
  valid_data = [(input_text, target) for input_text, target in zip(inputs, targets) if target and target != 'None']
  inputs = [input_text for input_text, _ in valid_data]
  targets = [target for _, target in valid_data]
  
  # Braille blank -> Regular blank
  targets = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in targets]
  return inputs, targets

def translate_text(text, model, tokenizer, max_length=256):
    model.to('cuda')
    # 모델의 사전학습 prompt와 일관성 유지
    input_text = f'translate Korean to Braille: {text}\nBraille:'
    inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, truncation=True, padding=True).to('cuda')

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    output = [token for token in outputs[0] if token not in [0, 1, -100]]
    return output

def eval(preds, targets):
    bleu_metric = load("google_bleu")
    wer_metric = load("wer")
    cer_metric = load("cer")
    length = len(preds)
    correct, wer, cer, bleu = 0, 0, 0, 0
    wrong_list = []
    for i, (pred, target) in enumerate(zip(preds, targets)):
        target = target[:len(pred)]
        pred = pred[:len(target)]
        wer_score = wer_metric.compute(predictions=[pred], references=[target])
        cer_score = cer_metric.compute(predictions=[pred], references=[target])
        bleu_score = bleu_metric.compute(predictions=[pred], references=[[target]])
        wer += wer_score
        cer += cer_score
        bleu += bleu_score['google_bleu']

        if wer_score == 0:
            correct += 1
        else:
            wrong_list.append(i+1)
            print(f"sample number: {i+1}")
            print(f"target: {target}")
            print(f"pred: {pred}")
            print(f"WER Score: {wer_score}")
            print(f"CER Score: {cer_score}")
            print(f"BLEU Score: {bleu_score['google_bleu']}")

    print(f"""
Correct: {correct}
Correct Rate: {correct/length}
Avg WER: {wer/length}
Avg CER: {cer/length}
Avg BLEU: {bleu/length}
Incorrect Sample: {wrong_list}
    """)
    return correct

def read_korean_braille_pairs(file_path):
    korean_list = []
    braille_list = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file if line.strip()]
            for i in range(0, len(lines), 2):
                korean = lines[i]
                braille = lines[i + 1]
                korean_list.append(korean)
                braille_list.append(braille)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except IndexError:
        print("Error: File format is incorrect. Ensure each Korean line is followed by its Braille counterpart.")

    korean_list = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in korean_list]
    braille_list = [elem.replace('⠀', ' ').replace('\u2800', ' ') for elem in braille_list]
    return korean_list, braille_list
