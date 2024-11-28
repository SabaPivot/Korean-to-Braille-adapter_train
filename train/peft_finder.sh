#!/bin/bash

# 지원하는 adapters
peft_models=("LoRA" "IA3" "prefix")

# iterate할 범위
declare -A lora_lr_epochs=(
  [3e-4]="4"
  [1e-4]="4 8 12"
  [9e-5]="4 8 12"
  [7e-5]="8 12 16"
  [5e-5]="12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 30 32 34 36"
  [3e-5]="16 20 24 28 32 36 40"
  [1e-5]="20 24 28 32 36 40 44 48 52 56 60 64"
)
ia3_lr=("5e-4" "3e-4")
ia3_epochs="10 15 20"
prefix_lr=("1e-4" "9e-5" "7e-5" "5e-5")
prefix_epochs="10 15 20"

# 스크립트 이름
TRAIN_SCRIPT="./train_adapters.sh"
BENCHMARK_SCRIPT="./validation_adapter_train.sh"

# 로깅 파일
LOG_FILE="peft_training.log"

echo "Starting PEFT parameter search..." > $LOG_FILE

# 훈련 실행
for peft_model in "${peft_models[@]}"; do
  if [ "$peft_model" == "LoRA" ]; then
    for lr in "${!lora_lr_epochs[@]}"; do
      for epoch in ${lora_lr_epochs[$lr]}; do
        echo "Testing $peft_model with LR=$lr and Epoch=$epoch..." | tee -a $LOG_FILE
        $TRAIN_SCRIPT "$peft_model" "$epoch" "$lr"
        if [ $? -ne 0 ]; then
          echo "Training failed for $peft_model with LR=$lr and Epoch=$epoch." | tee -a $LOG_FILE
          continue
        fi
        result=$($BENCHMARK_SCRIPT)
        exit_code=$?
        echo "Benchmark result: $result" | tee -a $LOG_FILE
        if [ $exit_code -eq 0 ] && echo "$result" | grep -q "All correct!"; then
          echo "Successful configuration: $peft_model, LR=$lr, Epoch=$epoch" | tee -a $LOG_FILE
          exit 0
        fi
      done
    done
  elif [ "$peft_model" == "IA3" ]; then
    for lr in "${ia3_lr[@]}"; do
      for epoch in $ia3_epochs; do
        echo "Testing $peft_model with LR=$lr and Epoch=$epoch..." | tee -a $LOG_FILE
        $TRAIN_SCRIPT "$peft_model" "$epoch" "$lr"
        if [ $? -ne 0 ]; then
          echo "Training failed for $peft_model with LR=$lr and Epoch=$epoch." | tee -a $LOG_FILE
          continue
        fi
        result=$($BENCHMARK_SCRIPT)
        exit_code=$?
        echo "Benchmark result: $result" | tee -a $LOG_FILE
        if [ $exit_code -eq 0 ] && echo "$result" | grep -q "All correct!"; then
          echo "Successful configuration: $peft_model, LR=$lr, Epoch=$epoch" | tee -a $LOG_FILE
          exit 0
        fi
      done
    done
  elif [ "$peft_model" == "prefix" ]; then
    for lr in "${prefix_lr[@]}"; do
      for epoch in $prefix_epochs; do
        echo "Testing $peft_model with LR=$lr and Epoch=$epoch..." | tee -a $LOG_FILE
        $TRAIN_SCRIPT "$peft_model" "$epoch" "$lr"
        if [ $? -ne 0 ]; then
          echo "Training failed for $peft_model with LR=$lr and Epoch=$epoch." | tee -a $LOG_FILE
          continue
        fi
        result=$($BENCHMARK_SCRIPT)
        exit_code=$?
        echo "Benchmark result: $result" | tee -a $LOG_FILE
        if [ $exit_code -eq 0 ] && echo "$result" | grep -q "All correct!"; then
          echo "Successful configuration: $peft_model, LR=$lr, Epoch=$epoch" | tee -a $LOG_FILE
          exit 0
        fi
      done
    done
  fi
done

# 성공적인 PEFT 모델을 찾으면 (validate에서 모두 맞으면) 훈련 종료, configuration 출력
echo "All configurations tested. No successful configuration found." | tee -a $LOG_FILE
exit 1
