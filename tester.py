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

        # [핵심 수정] 클래스 이름 매핑 파일 로드
        self.class_names = []
        # .h5 파일 경로에서 _classes.json 파일 경로를 추론
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
        단일 영상 프레임을 처리하여 행동을 예측하고 시각화된 프레임을 반환
        """
        frame_resized = letterbox_resize(frame, self.resize_dims)
        display_frame = frame_resized.copy()

        results = self.yolo_model(frame_resized, verbose=False, conf=0.5, device=self.device)

        person_detected = results[0].keypoints is not None and len(results[0].keypoints.xy) > 0

        if person_detected:
            all_boxes = results[0].boxes.xyxy.cpu().numpy()
            box_areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
            target_index = np.argmax(box_areas)

            raw_kp = results[0].keypoints.xy.cpu().numpy()[target_index]
            conf = results[0].keypoints.conf.cpu().numpy()[target_index]
            box = results[0].boxes.xyxy.cpu().numpy()[target_index]

            filled_kp = fill_missing_keypoints(raw_kp, conf, self.last_valid_pose)
            self.last_valid_pose = filled_kp
            anchor = get_stable_anchor(filled_kp, conf)
            normalized_data = normalize_to_relative(filled_kp, anchor)

            self.buffer.append(normalized_data)

            if len(self.buffer) == self.seq_length:
                sequence_data = np.array(self.buffer)
                input_data = np.expand_dims(sequence_data, axis=0)

                prediction_probs = self.lstm_model.predict(input_data, verbose=0)[0]
                predicted_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[predicted_idx]

                # [핵심 수정] 예측된 인덱스를 실제 클래스 이름으로 변환
                predicted_class_name = f"Class #{predicted_idx}"  # 기본값 (매핑 파일 없을 시)
                if self.class_names and predicted_idx < len(self.class_names):
                    # 매핑 파일이 있고, 인덱스가 범위 내에 있으면 이름으로 변환
                    predicted_class_name = self.class_names[predicted_idx]

                self.current_prediction = f"{predicted_class_name.upper()} ({confidence:.2f})"

            self._draw_visualization(display_frame, raw_kp, conf, box)
        else:
            self.buffer.clear()
            self.current_prediction = "No Person Detected"

        label_color = (0, 255, 255)  # 노란색으로 통일

        cv2.putText(display_frame, f"Prediction: {self.current_prediction}", (10, 20),
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