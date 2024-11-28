# T5 모델 few-shot PEFT(Parameter-Efficient Fine-Tuning) 학습 프레임워크
본 레포지터리는 T5 모델에 few-shot PEFT 학습을 적용하는 프레임워크입니다. 본 레포지터리는 few-shot ICL(In-Context Learning)을 적용하기 어려운 Seq2Seq 모델에 대해서 few-shot PEFT가 효과적이라는 [neurips에 게재된 논문](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html)의 주장을 근거로 합니다.

## Repository 구조
```
t5-adapter/
├── train/                          # 어댑터 학습 관련 스크립트
│   ├── train_adapters.py           # 어댑터 학습 메인 스크립트
│   ├── train_adapters.sh           # 어댑터 학습 쉘 스크립트
│   ├── validation_adapter_train.py # 어댑터 검증 메인 스크립트
│   ├── validation_adapter_train.sh # 어댑터 검증 쉘 스크립트
│   ├── peft_finder.sh              # 어댑터 학습과 검증으로 최적 파라미터 탐색 쉘 스크립트
│   └── fewshot-example.txt         # 어댑터 훈련에 사용한 few-shot 예시 데이터
├── success_adapter/                # 학습된 어댑터 모델
│   ├── brackets/                   # 괄호 기호 어댑터
│   └── euro_dollar_yen/            # 화폐 기호 어댑터
├── benchmark_peft.py               # 벤치마크 평가 스크립트
├── benchmark_peft.sh               # 벤치마크 쉘 스크립트
├── utils.py                        # 재사용성 유틸리티 함수
└── requirements.txt                # 필수 패키지 리스트
```

## 주요 기능
- LoRA, IA3, prefix-tuning 어댑터 훈련
- validation data를 사용하여 어댑터 훈련 최적 파라미터 탐색
- 어댑터 사용 벤치마크 평가`
- 어댑터 훈련 과정 로깅
## 실행 방법
### 필수 패키지 설치
```
pip install -r requirements.txt
```
### 어댑터 훈련
1. 어댑터 훈련에 사용할 모델을 `train_adapters.sh`와 `validation_adapter_train.sh` 스크립트의 `--model_name` 인자에 입력합니다. 특정할 버전이 있다면 `revision`에 명시합니다.
2. `fewshot-example.txt` 파일에 어댑터 훈련에 사용할 few-shot 예시를 작성합니다.
3. validation set으로 사용할 데이터의 경로를 `validation_adapter_train.sh`의 `--validation_data_path` 인자에 입력합니다.
4. `peft_finder.sh` 스크립트를 실행하여 최적의 어댑터 훈련 파라미터를 탐색합니다. 탐색하기 원하는 어댑터 종류와 learning rate, 학습 epoch 수를 수정할 수 있습니다. 현재 학습을 제공하는 어댑터는 `LoRA`, `IA3`, `prefix-tuning` 세 종류입니다.
5. 모든 learning rate, 학습 epoch으로 훈련이 완료되거나, 검증 과정에서 `inputs`과 `targets`이 완전히 일치하는 경우 훈련이 종료됩니다.
6. 훈련이 종료되면 `results/` 폴더가 생성되며 훈련된 어댑터가 저장됩니다.
7. 성공적으로 훈련된 어댑터는 `success_adapter/` 폴더에 보관하기를 추천합니다.
8. 훈련의 모든 과정과 각 파라미터의 validation 점수는 `peft_training.log`에 기록됩니다.


### 훈련 스크립트 실행
```
chmod +x train/train_adapters.sh
chmod +x train/validation_adapter_train.sh
chmod +x train/peft_finder.sh
./train/peft_finder.sh
```

### 어댑터 사용 벤치마크 평가
*주의 사항*:

**어댑터는 훈련된 문자들에 한해 적용됩니다. 예를 들어, 화폐 기호를 예시로 학습한 어댑터는 반드시 화폐 기호가 input으로 전달되었을 때 호출됩니다. 코드로 명시된 문자들 외에 다른 문자들에 대해서는 어댑터가 적용되지 않고, 기본 모델의 점역 결과가 호출됩니다.** 
1. 훈련된 어댑터를 `success_adapter/` 폴더에 저장합니다.
2. 평가에 사용할 벤치마크 데이터 경로를 `benchmark_peft.sh`의 `--benchmark_data_path` 인자에 입력합니다.
3. `benchmark_peft.sh`의 코드를 수정합니다.

예시) 

line 27-31:
```
special_characters = {'<', '>', '《', '》', '『', '』'} 
special_characters_path = os.path.join(peft_model_path, 'brackets/LoRA, LR=5e-5, Epoch=19')

currency_character = {'$', '￥', '€'}
currency_character_path = os.path.join(peft_model_path, 'euro_dollar_yen/LoRA, LR=5e-5, Epoch=30')

# 추가로 훈련한 어댑터를 사용하고 싶다면 다음 코드를 추가합니다.
custom_characters = {...}
custom_characters_path = os.path.join(peft_model_path, 'custom_characters/adapter configuartion')
```
line 36-40:
```
    if any(char in special_characters for char in text):
        model = PeftModel.from_pretrained(model, special_characters_path)

    elif any(char in currency_character for char in text):
        model = PeftModel.from_pretrained(model, currency_character_path)

    # 추가로 훈련한 어댑터를 사용하고 싶다면 다음 코드를 추가합니다.
    elif any(char in custom_characters for char in text):
        model = PeftModel.from_pretrained(model, custom_characters_path)
```
3. `benchmark_peft.sh` 스크립트를 실행합니다.
4. 검증 결과로 `BLEU`, `WER`, `CER` 메트릭이 출력됩니다.

### 어댑터 사용 벤치마크 평가 스크립트 실행
```
chmod +x benchmark_peft.sh
./benchmark_peft.sh
```

## 예시 결과
`success_adapter/` 디렉토리에는 사전에 훈련된 어댑터 두 종류가 저장되어 있습니다. 각 어댑터는 화폐 기호와 괄호 기호에 대해 훈련되었습니다.
- `brackets/LoRA, LR=5e-5, Epoch=19`: 괄호 기호 어댑터
- `euro_dollar_yen/LoRA, LR=5e-5, Epoch=30`: 화폐 기호 어댑터
- 베이스 모델 `VEXFZU/t5-xlarge-ko-kb`은 각각 화폐 기호와 괄호 기호에 대해 훈련되지 않았습니다. 따라서, 해당 단어들을 올바르게 점역하지 못합니다.

### 예시 validation set (`validation_data.json`)
| Model                        | Correct | BLEU  |
|------------------------------|---------|-------|
| PEFT model with adapter      | 3       | 1.0   |
| Base model (`VEXFZU/t5-xlarge-ko-kb`) | 0       | 0.0   |

### 벤치마크 평가 결과
**본 벤치마크 평가는 국가 공인 점역 교정사 자격 시험 159문제를 사용하였습니다.**
| Metric        | With Adapter       | Without Adapter    |
|---------------|--------------------|--------------------|
| Correct       | 141                | 136                |
| Correct Rate  | 0.8867 | 0.8553 |
| WER           | 0.0343 | 0.0422 |
| CER           | 0.0117 | 0.0139 |
| BLEU          | 0.9501 | 0.9383 |


## 참고 문헌
- [Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html)
- [Huggingface PEFT documentation](https://huggingface.co/docs/transformers/main/peft)
