from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer
import json
from datasets import Dataset
import torch
from peft import AutoPeftModelForSeq2SeqLM, get_peft_model, TaskType, LoraConfig, IA3Config, PrefixEncoder, PrefixTuningConfig, PromptEncoderConfig
import argparse
from ..utils import read_korean_braille_pairs, preprocess_function

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="VEXFZU/t5-xlarge-ko-kb", required=True, help="Path to the model checkpoint.")
parser.add_argument("--revision", type=str, default="main", required=False, help="Revision name for the checkpoint.")
parser.add_argument("--peft_model", type=str, default="LoRA")
parser.add_argument("--num_train_epoch", type=int)
parser.add_argument("--lr", type=float)
parser.add_argument("--file_path", type=str, default="fewshot-example.txt")

args = parser.parse_args()

model_name = args.model_name
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, revision = args.revision)
tokenizer = AutoTokenizer.from_pretrained(model_name)
fewshot_path = args.file_path

inputs, targets = read_korean_braille_pairs(fewshot_path)
test_data = {"source": inputs, "target": targets}

dataset = Dataset.from_dict(test_data)
processed_dataset = dataset.map(
    lambda examples: preprocess_function(examples, tokenizer, source_lang="Korean", target_lang="Braille"),
    batched=True
)

processed_dataset = processed_dataset.remove_columns(["source", "target"])

if args.peft_model == "LoRA":
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        target_modules=["q", "v"],
        task_type=TaskType.SEQ_2_SEQ_LM,
        lora_alpha=32,
        lora_dropout=0.05
    )
    model = get_peft_model(model, lora_config)

if args.peft_model == "IA3":
    # IA3 config
    IA3_config = IA3Config(
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, IA3_config)

if args.peft_model == "prefix":
    # prefix prompt config
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        num_virtual_tokens=20
        )
    model = get_peft_model(model, prefix_config)

model.print_trainable_parameters()

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=args.num_train_epoch,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no",
    learning_rate=args.lr,
    gradient_checkpointing=False,
    optim="adamw_hf",
    warmup_ratio=0.03,
    weight_decay=0.01,
    max_grad_norm=0.3,
    remove_unused_columns=False,
    bf16=True,
    report_to="none",
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

trainer.train()

model.save_pretrained("results")
