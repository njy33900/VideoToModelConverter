import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import torch
import os
import json

# 기존 유틸리티 재사용
from pose_utils import fill_missing_keypoints, get_stable_anchor, normalize_to_relative, letterbox_resize


class VideoTester:
    """
    영상 프레임을 실시간으로 받아 YOLO와 LSTM 모델을 통해 행동을 예측하는 클래스
    """

    def __init__(self, yolo_model_path: str, lstm_model_path: str):
        """
        VideoTester를 초기화

        Args:
            yolo_model_path (str): YOLO 모델 파일(.pt) 경로
            lstm_model_path (str): 학습된 LSTM 모델 파일(.h5) 경로
        """
        # 설정값
        self.seq_length = 30
        self.resize_dims = (320, 240)

        # 시계열 데이터 버퍼 (최대 30 프레임 저장)
        self.buffer = deque(maxlen=self.seq_length)
        self.last_valid_pose = np.zeros((17, 2))

        # GPU 설정
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Tester Device: {self.device}")

        # 모델 로딩
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        if self.device == '0':
            self.yolo_model.to('cuda')

        print(f"Loading LSTM model: {lstm_model_path}")
        self.lstm_model = tf.keras.models.load_model(lstm_model_path)

        self.class_names = []
        class_map_path = lstm_model_path.replace('.h5', '_classes.json')

        if os.path.exists(class_map_path):
            try:
                with open(class_map_path, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
                print(f"클래스 맵 로드 성공: {len(self.class_names)}개 클래스 -> {self.class_names}")
            except Exception as e:
                print(f"경고: 클래스 맵 파일({class_map_path}) 로드 실패: {e}")
        else:
            print(f"경고: 클래스 맵 파일({class_map_path})을 찾을 수 없습니다. 인덱스 번호로 표시됩니다.")

        # 현재 예측 결과 저장 변수
        self.current_prediction = "Initializing..."

        # 시각화용 뼈대 연결 정보
        self.skeleton = [
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]

    def process_frame(self, frame: np.ndarray) -> (str, np.ndarray):
        """
        학습 파이프라인과 동일하게 위치, 속도, 가속도 특징을 생성
        """
        frame_resized = letterbox_resize(frame, self.resize_dims)
        display_frame = frame_resized.copy()

        results = self.yolo_model(frame_resized, verbose=False, conf=0.5, device=self.device)
        person_detected = results[0].keypoints is not None and len(results[0].keypoints.xy) > 0

        norm_kp = np.zeros(34)  # 기본값
        if person_detected:
            all_boxes = results[0].boxes.xyxy.cpu().numpy()
            target_index = np.argmax((all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1]))

            raw_kp = results[0].keypoints.xy.cpu().numpy()[target_index]
            conf = results[0].keypoints.conf.cpu().numpy()[target_index]
            box = results[0].boxes.xyxy.cpu().numpy()[target_index]

            filled_kp = fill_missing_keypoints(raw_kp, conf, self.last_valid_pose)
            self.last_valid_pose = filled_kp
            anchor = get_stable_anchor(filled_kp, conf)

            norm_kp = normalize_to_relative(filled_kp, anchor)
            self._draw_visualization(display_frame, raw_kp, conf, box)

        self.buffer.append(norm_kp)

        # 버퍼가 30개로 꽉 찼을 때만 예측을 수행
        if len(self.buffer) == self.seq_length:
            # --- converter.py와 동일한 특징 공학 로직 ---
            # 위치 데이터 (30, 34)
            pos_seq = np.array(self.buffer)

            # 속도 데이터 (29, 34)
            vel_seq = np.diff(pos_seq, axis=0)

            # 가속도 데이터 (28, 34)
            acc_seq = np.diff(vel_seq, axis=0)

            # 길이를 맞추기 위해 패딩 추가
            vel_seq_padded = np.pad(vel_seq, ((1, 0), (0, 0)), 'constant')
            acc_seq_padded = np.pad(acc_seq, ((2, 0), (0, 0)), 'constant')

            # 모든 특징을 결합하여 (30, 102) 형태로 만듦
            full_feature_seq = np.concatenate((pos_seq, vel_seq_padded, acc_seq_padded), axis=1)
            # -----------------------------------------

            # 모델 입력 형태에 맞게 reshape: (1, 30, 102)
            input_data = np.expand_dims(full_feature_seq, axis=0)

            prediction_probs = self.lstm_model.predict(input_data, verbose=0)[0]
            top_3_indices = np.argsort(prediction_probs)[::-1][:3]

            top_predictions = []
            for i in top_3_indices:
                class_name = f"Class #{i}"
                if self.class_names and i < len(self.class_names):
                    class_name = self.class_names[i]
                confidence = prediction_probs[i]
                top_predictions.append(f"{class_name.upper()}: {confidence:.0%}")

            self.current_prediction = "\n".join(top_predictions)

        # 예측 텍스트를 이미지에 오버레이
        first_line_prediction = self.current_prediction.split('\n')[0]
        label_color = (0, 255, 255)

        cv2.putText(display_frame, f"Prediction: {first_line_prediction}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2, cv2.LINE_AA)

        return self.current_prediction, display_frame

    def _draw_visualization(self, img, kp, conf, box):
        """ 프레임 위에 객체 박스, 뼈대, 관절을 표시 """
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

        for s, e in self.skeleton:
            if conf[s] > 0.5 and conf[e] > 0.5:
                cv2.line(img, (int(kp[s][0]), int(kp[s][1])), (int(kp[e][0]), int(kp[e][1])),
                         (255, 255, 255), 2)

        for i, (x, y) in enumerate(kp):
            if conf[i] > 0.5:
                color = (255, 0, 0)  # 파란색
                cv2.circle(img, (int(x), int(y)), 5, color, -1)