# lstm 모델 학습 코드

import numpy as np
import os
import datetime
from keras.models import Sequential, load_model # 🌟 load_model 추가
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import StratifiedShuffleSplit

# --- ⚙️ 설정: 이 스위치로 모드를 변경하세요 ---
# True: model_save_path에 있는 모델을 불러와서 이어서 학습합니다.
# False: 기존처럼 새로운 모델을 처음부터 학습합니다.
CONTINUE_TRAINING = False
# -----------------------------------------

def train_lstm_model_dual(X_path, y_path, model_save_path):
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"🔹 X shape: {X.shape}")
    print(f"🔹 y shape: {y.shape}")

    y_classes = np.argmax(y, axis=1)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X, y_classes))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # --- 👇 여기가 핵심 변경 부분입니다 👇 ---
    if CONTINUE_TRAINING and os.path.exists(model_save_path):
        print(f"📖 기존 모델 '{model_save_path}'을(를) 불러와 이어서 학습합니다.")
        model = load_model(model_save_path)
    else:
        print("✨ 새로운 모델을 처음부터 학습합니다.")
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
            LSTM(64, return_sequences=False),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(y.shape[1], activation='softmax')
        ])
    # --- 👆 여기까지가 핵심 변경 부분입니다 👆 ---

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', save_best_only=True, verbose=1, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='min')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    print("▶ 모델 학습을 시작합니다.")
    model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard_callback]
    )

    print(f"✅ 양손 LSTM 모델 학습 및 저장 완료: {model_save_path}")

if __name__ == "__main__":
    train_lstm_model_dual(
        X_path="processed_lstm/X_seq_lstm_dual.npy",
        y_path="processed_lstm/y_seq_lstm_dual.npy",
        # 🌟 불러오고 저장할 모델 파일 경로를 정확히 지정해야 합니다.
        model_save_path="models/gesture_lstm_model_dual_v2.h5" 
    )