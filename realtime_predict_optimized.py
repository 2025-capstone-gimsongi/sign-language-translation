import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import joblib
from collections import deque
from PIL import ImageFont, ImageDraw, Image
import threading

# --- 설정값 ---
MODEL_PATH = "models/gesture_lstm_model_dual_v2.h5" 
ENCODER_PATH = "processed_lstm/label_encoder_lstm_dual.pkl"
FRAMES_PER_SEQUENCE = 30 
FONT_PATH = "C:/Windows/Fonts/malgun.ttf" # 사용자 환경에 맞는 한글 폰트 경로
CONFIDENCE_THRESHOLD = 0.8 # 이 값 이상의 확신도를 가질 때 초록색으로 표시

# --- 스레딩 관련 전역 변수 ---
prediction_result = ("", 0.0)
is_predicting = False # 현재 예측 스레드가 실행 중인지 확인하는 플래그

# --- 모델 및 리소스 로드 ---
print("▶ 모델과 리소스를 로드하는 중입니다...")
model = load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
print("✅ 모델 로드 완료!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

frame_buffer = deque(maxlen=FRAMES_PER_SEQUENCE)

# --- 한글 텍스트 출력 함수 ---
def draw_korean_text(img, text, position, font_size=32, color=(255, 255, 255)):
    font = ImageFont.truetype(FONT_PATH, font_size)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# --- 별도 스레드에서 모델 예측을 수행하는 함수 ---
def predict_gesture(sequence_data):
    global prediction_result, is_predicting
    
    # 1. 모델 예측 수행
    prediction = model.predict(sequence_data, verbose=0)
    
    # 2. 확신도와 예측 라벨을 항상 가져옴
    confidence = np.max(prediction)
    predicted_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    # 3. 확신도와 상관없이 예측 결과 업데이트
    prediction_result = (predicted_label, confidence)
        
    # 4. 예측이 끝났음을 알림
    is_predicting = False

# --- 메인 루프 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

print("▶ 실시간 성능 확인을 시작합니다. ('q'를 누르면 종료)")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # --- 데이터 정규화 및 버퍼 추가 ---
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_data = {"Left": [], "Right": []}
        
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = hand_handedness.classification[0].label
            hand_data[hand_label] = [lm for lm in hand_landmarks.landmark]

        normalized_left = [0.0] * 63
        if hand_data["Left"]:
            left_wrist = hand_data["Left"][0]
            for i, lm in enumerate(hand_data["Left"]):
                normalized_left[i*3] = lm.x - left_wrist.x
                normalized_left[i*3 + 1] = lm.y - left_wrist.y
                normalized_left[i*3 + 2] = lm.z - left_wrist.z

        normalized_right = [0.0] * 63
        if hand_data["Right"]:
            right_wrist = hand_data["Right"][0]
            for i, lm in enumerate(hand_data["Right"]):
                normalized_right[i*3] = lm.x - right_wrist.x
                normalized_right[i*3 + 1] = lm.y - right_wrist.y
                normalized_right[i*3 + 2] = lm.z - right_wrist.z
        
        one_frame = normalized_left + normalized_right
        frame_buffer.append(one_frame)
    else:
        frame_buffer.clear()

    # --- 예측 스레드 실행 ---
    frame_count += 1
    if len(frame_buffer) == FRAMES_PER_SEQUENCE and not is_predicting and frame_count % 5 == 0:
        is_predicting = True
        sequence_data = np.array(frame_buffer).reshape(1, FRAMES_PER_SEQUENCE, 126)
        
        thread = threading.Thread(target=predict_gesture, args=(sequence_data,))
        thread.start()

    # --- 화면 출력 ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
    label, conf = prediction_result
    status_text = f"Prediction: {label} ({conf:.2f})"
    color = (0, 255, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 0, 255)
    
    frame = draw_korean_text(frame, status_text, (10, 50), font_size=40, color=color)
    
    cv2.imshow("Optimized Performance Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()