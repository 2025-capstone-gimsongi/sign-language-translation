# 수집된 데이터 전처리 코드

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import joblib

def preprocess_dual_hand_csv_lstm(csv_folder, save_dir):
    all_X, all_y = [], []

    for file in os.listdir(csv_folder):
        if file.endswith(".csv"):
            label = file.replace("_sequences.csv", "").replace(".csv", "")
            file_path = os.path.join(csv_folder, file)
            df = None # DataFrame 초기화

            # --- ✅ 이중 인코딩 처리 로직 시작 ---
            try:
                # 1. UTF-8로 먼저 시도. pandas가 헤더를 자동으로 감지하도록 함.
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                # 2. UTF-8 실패 시, CP949로 재시도
                try:
                    print(f"⚠️ INFO: UTF-8 decoding failed for '{file}'. Retrying with CP949...")
                    df = pd.read_csv(file_path, encoding='cp949')
                except Exception as e:
                    print(f"❌ ERROR: Failed to read '{file}' with both UTF-8 and CP949. Skipping. Error: {e}")
                    continue # 다음 파일로 넘어감
            except Exception as e:
                 print(f"❌ ERROR: An unexpected error occurred while reading '{file}'. Skipping. Error: {e}")
                 continue
            # --- ✅ 이중 인코딩 처리 로직 끝 ---

            # label이라는 이름의 열이 있다면 제거 (헤더가 있는 경우 대비)
            if 'label' in df.columns:
                X = df.drop('label', axis=1).values.astype(np.float32)
            else:
                # 헤더가 없는 경우, 마지막 열이 라벨일 수 있으나 여기서는 모든 열을 데이터로 간주
                # (데이터 수집 방식에 따라 조정 필요)
                X = df.iloc[:, :-1].values.astype(np.float32)

            if X.shape[1] == 0:
                print(f"WARNING: No data in '{file}'. Skipping.")
                continue
            
            if X.shape[1] != 3780:
                 print(f"WARNING: Column count mismatch in '{file}' (is {X.shape[1]}, expected 3780). Skipping.")
                 continue

            X = X.reshape(-len(df), 30, 126)
            y = [label] * len(X)
            all_X.append(X)
            all_y.extend(y)

    if not all_X:
        print("❌ No data was processed. Check the 'data' folder and error messages.")
        return

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.array(all_y)

    # (이하 LabelEncoder 및 저장 로직은 동일)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_all)
    y_onehot = to_categorical(y_encoded)

    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_seq_lstm_dual.npy"), X_all)
    np.save(os.path.join(save_dir, "y_seq_lstm_dual.npy"), y_onehot)
    joblib.dump(le, os.path.join(save_dir, "label_encoder_lstm_dual.pkl"))

    print(f"✅ Preprocessing complete: {X_all.shape[0]} samples processed.")


if __name__ == "__main__":
    preprocess_dual_hand_csv_lstm(
        csv_folder="data",
        save_dir="processed_lstm"
    )