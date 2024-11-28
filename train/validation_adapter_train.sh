#!/bin/bash

output=$(python validation_adapter_train.py \
  --model_name "VEXFZU/t5-xlarge-ko-kb" \
  --peft_model "/results")

# validaton_adapter_train.py의 exit code 확인
if [ $? -eq 0 ]; then
    echo "$output" 
else
    echo "$output"
fi
