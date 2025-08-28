# 수어 인식용 카메라 + 인식 모델 사용 코드

import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# --- 설정값 ---
# 🌟 v2 모델 경로로 수정했습니다.
MODEL_PATH = "models/gesture_lstm_model_dual_v2.h5" 
ENCODER_PATH = "processed_lstm/label_encoder_lstm_dual.pkl"
# 🌟 학습 시 사용한 프레임 수와 동일하게 맞춰주세요.
FRAMES_PER_SEQUENCE = 30 
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 사용자 환경에 맞는 한글 폰트 경로로 설정
CONFIDENCE_THRESHOLD = 0.8  # 이 값 이상의 확신도를 가질 때만 결과를 표시

# --- 모델 및 리소스 로드 ---
print("▶ 모델과 리소스를 로드하는 중입니다...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("✅ 모델 로드 완료!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 시퀀스 데이터를 저장할 deque 생성
frame_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)

# --- 한글 텍스트 출력을 위한 함수 ---
def draw_korean_text(img, text, position, font_size=32, color=(255, 255, 255)):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 메인 루프 ---
cap = cv2.VideoCapture(0)
print("▶ 실시간 성능 확인을 시작합니다. ('q'를 누르면 종료)")

prediction_result = ("", 0.0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- 🌟 데이터 수집과 동일한 정규화 로직 적용 ---
    hand_data = {"Left": [0.0] * 63, "Right": [0.0] * 63}
    hand_detected = {"Left": False, "Right": False}

    if results.multi_hand_landmarks:
        # 1. 랜드마크 절대 좌표 수집
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            coords = [lm for lm in hand_landmarks.landmark]
            hand_data[hand_label] = coords
            hand_detected[hand_label] = True
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # 2. 손목 기준 상대 좌표로 정규화
        normalized_left = [0.0] * 63
        if hand_detected["Left"]:
            left_wrist = hand_data["Left"][0]
            for i, lm in enumerate(hand_data["Left"]):
                normalized_left[i*3] = lm.x - left_wrist.x
                normalized_left[i*3 + 1] = lm.y - left_wrist.y
                normalized_left[i*3 + 2] = lm.z - left_wrist.z

        normalized_right = [0.0] * 63
        if hand_detected["Right"]:
            right_wrist = hand_data["Right"][0]
            for i, lm in enumerate(hand_data["Right"]):
                normalized_right[i*3] = lm.x - right_wrist.x
                normalized_right[i*3 + 1] = lm.y - right_wrist.y
                normalized_right[i*3 + 2] = lm.z - right_wrist.z
        
        # 3. 정규화된 좌표를 하나의 프레임으로 합치기
        one_frame = normalized_left + normalized_right
        frame_buffer.append(one_frame)
    else:
        # 손이 감지되지 않으면 버퍼를 비워 잘못된 예측 방지
        frame_buffer.clear()

    # --- 모델 예측 ---
    if len(frame_buffer) == FRAMES_PER_SEQUENCE:
        # 시퀀스 데이터를 모델 입력 형태에 맞게 변환
        sequence_data = np.array(frame_buffer).reshape(1, FRAMES_PER_SEQUENCE, 126)
        
        # 예측 실행
        prediction = model.predict(sequence_data, verbose=0)
        confidence = np.max(prediction)
        
        if confidence >= CONFIDENCE_THRESHOLD:
            predicted_index = np.argmax(prediction)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
            prediction_result = (predicted_label, confidence)
        else:
            prediction_result = ("?", 0.0) # 확신도가 낮으면 '?'로 표시

    # --- 화면에 결과 출력 ---
    label, conf = prediction_result
    status_text = f"Prediction: {label} ({conf:.2f})"
    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 0, 255) # 확신도 높으면 초록, 낮으면 빨강
    
    frame = draw_korean_text(frame, status_text, (10, 50), font_size=40, color=color)
    
    cv2.imshow("LSTM Model Performance Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()