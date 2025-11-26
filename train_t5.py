import json
from datasets import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer,
    TrainingArguments,
    Trainer
)
import numpy as np
import os

# 모델 불러오기

MODEL_NAME = "paust/pko-t5-base"
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

DATASET_PATH = "./train_dataset.json"
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 데이터를 훈련용과 검증용으로 분리
np.random.shuffle(raw_data)
split_point = int(len(raw_data) * 0.9)
train_data = raw_data[:split_point]
eval_data = raw_data[split_point:]

print(f"훈련 데이터: {len(train_data)}개, 검증 데이터: {len(eval_data)}개")

train_dataset = Dataset.from_list(train_data)
eval_dataset = Dataset.from_list(eval_data)

# 데이터 전처리
def preprocess_function(examples):
    inputs = tokenizer(examples["input"], padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(examples["output"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 모델 학습 설정값

MAX_EPOCHS = 10

training_args = TrainingArguments(
    output_dir="./t5_finetune_results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_steps=50,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset
)

best_eval_loss = float("inf")
patience_counter = 0
early_stopping_patience = 3 # n번 연속 성능 향상이 없으면 중단
SAVE_PATH = "./my_finetuned_t5_model"

for epoch in range(MAX_EPOCHS):
    print(f"--- Epoch {epoch + 1}/{MAX_EPOCHS} ---")
    
    trainer.train()
    
    # 모델 평가
    eval_results = trainer.evaluate()
    current_eval_loss = eval_results["eval_loss"]
    print(f"Epoch {epoch + 1} - 검증 손실(Validation Loss): {current_eval_loss}")
    
    if current_eval_loss < best_eval_loss:
        best_eval_loss = current_eval_loss
        patience_counter = 0
        print(f"최고 성능 모델 발견 (손실: {best_eval_loss:.4f})")
        print(f"모델을 '{SAVE_PATH}'에 저장합니다")
        trainer.save_model(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
    else:
        patience_counter += 1
        print(f"성능 향상 없음. (Patience: {patience_counter}/{early_stopping_patience})")
    
    # 조기 종료 조건 확인
    if patience_counter >= early_stopping_patience:
        print(f"{early_stopping_patience}번 연속 성능 미향상으로 학습 종료")
        break

print("✅ 모델 학습 완료")
