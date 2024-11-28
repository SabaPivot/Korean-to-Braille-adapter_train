#!/bin/bash

# Supporting adapters: LoRA, IA3, prefix

# Use codes below to start train with single set of parameters
# python train_adapters.py \
#   --model_name "VEXFZU/t5-xlarge-ko-kb" \
#   --revision "main" \
#   --peft_model "LoRA" \
#   --num_train_epoch 20 \
#   --lr 5e-5 \

# Use codes below to start train and edit peft_finder.sh to change sets of parameters
echo $1 $2 $3
python train_adapters.py \
  --model_name "VEXFZU/t5-xlarge-ko-kb" \
  --revision "main" \
  --peft_model $1 \
  --num_train_epoch $2 \
  --lr $3 \