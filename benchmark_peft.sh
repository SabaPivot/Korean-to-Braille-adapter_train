#!/bin/bash

python benchmark_peft.py \
  --model_name "VEXFZU/t5-xlarge-ko-kb" \
  --peft_model_path "/success_adapter"
