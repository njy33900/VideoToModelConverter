import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import deque
import torch
import os
import json
import time

from pose_utils import fill_missing_keypoints, get_stable_anchor, normalize_to_relative, letterbox_resize
from trainer.train_logic_transformer import PositionalEncoding


class VideoTester:
    """
    영상 프레임을 실시간으로 받아 YOLO와 LSTM 모델을 통해 행동을 예측하는 클래스
    (Time Sync, Tracking, Buffering Visualization 적용)
    """

    def __init__(self, yolo_model_path: str, lstm_model_path: str):
        # 설정값
        self.seq_length = 30
        self.resize_dims = (640, 640)

        # 시간 동기화 설정 (10 FPS)
        self.target_fps = 10
        self.interval = 1.0 / self.target_fps
        self.last_sampling_time = 0

        # 예측 부하 조절
        self.prediction_interval = 1
        self.sampling_counter = 0

        # 데이터 버퍼
        self.buffer = deque(maxlen=self.seq_length)
        self.last_valid_pose = None

        # 객체 추적용 ID
        self.target_id = None

        # GPU 설정
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Tester Device: {self.device}")

        # 모델 로딩
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        if self.device == '0':
            self.yolo_model.to('cuda')

        print(f"Loading Keras model: {lstm_model_path}")
        try:
            custom_objects = {'PositionalEncoding': PositionalEncoding}
            self.lstm_model = tf.keras.models.load_model(lstm_model_path, custom_objects=custom_objects)
        except ValueError:
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)

        self.class_names = []
        class_map_path = lstm_model_path.replace('.h5', '_classes.json')
        if os.path.exists(class_map_path):
            with open(class_map_path, 'r', encoding='utf-8') as f:
                self.class_names = json.load(f)

        # 초기 상태 텍스트 변경
        self.current_prediction = "System Ready. Gathering Data..."

        self.skeleton = [
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]

    def process_frame(self, frame: np.ndarray) -> (str, np.ndarray):
        current_time = time.time()

        # 리사이즈
        frame_resized = letterbox_resize(frame, self.resize_dims)
        display_frame = frame_resized.copy()

        # YOLO Tracking
        results = self.yolo_model.track(frame_resized, persist=True, verbose=False, conf=0.5, device=self.device,
                                        tracker="bytetrack.yaml")

        current_kp = None
        current_conf = None
        current_box = None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()

            if self.target_id is None:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_idx = np.argmax(areas)
                self.target_id = track_ids[max_idx]

            if self.target_id in track_ids:
                idx = track_ids.index(self.target_id)
                current_kp = keypoints[idx]
                current_conf = confs[idx]
                current_box = boxes[idx]
                self._draw_visualization(display_frame, current_kp, current_conf, current_box)
            else:
                self.target_id = None

        # LSTM 데이터 샘플링
        if current_time - self.last_sampling_time >= self.interval:
            self.last_sampling_time = current_time

            norm_kp = np.zeros(34)
            if current_kp is not None:
                if self.last_valid_pose is None: self.last_valid_pose = current_kp
                filled_kp = fill_missing_keypoints(current_kp, current_conf, self.last_valid_pose)
                self.last_valid_pose = filled_kp

                anchor = get_stable_anchor(filled_kp, current_conf)
                if anchor is None:
                    cx = (current_box[0] + current_box[2]) / 2
                    cy = (current_box[1] + current_box[3]) / 2
                    anchor = np.array([cx, cy])

                box_h = current_box[3] - current_box[1]
                scale_factor = max(box_h, 1.0)
                relative_kp = filled_kp - anchor
                norm_kp_vec = relative_kp / scale_factor
                norm_kp = norm_kp_vec.flatten()

            self.buffer.append(norm_kp)
            self.sampling_counter += 1

            # 버퍼가 아직 안 찼을 때 진행 상황 표시
            if len(self.buffer) < self.seq_length:
                self.current_prediction = f"Buffering... {len(self.buffer)}/{self.seq_length}"

            # 버퍼가 찼을 때 예측 수행
            elif self.sampling_counter % self.prediction_interval == 0:
                self._run_prediction()

        self._draw_overlay(display_frame)

        return self.current_prediction, display_frame

    def _run_prediction(self):
        pos_seq = np.array(self.buffer)
        vel_seq = np.diff(pos_seq, axis=0)
        acc_seq = np.diff(vel_seq, axis=0)

        vel_seq_padded = np.pad(vel_seq, ((1, 0), (0, 0)), 'constant')
        acc_seq_padded = np.pad(acc_seq, ((2, 0), (0, 0)), 'constant')

        full_feature_seq = np.concatenate((pos_seq, vel_seq_padded, acc_seq_padded), axis=1)
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

    def _draw_overlay(self, img):
        first_line = self.current_prediction.split('\n')[0]
        # Buffering 중일 때는 노란색, 완료되면 초록색 계열
        color = (0, 255, 255) if "Buffering" in first_line else (0, 255, 0)

        cv2.putText(img, f"Status: {first_line}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if self.target_id is not None:
            cv2.putText(img, f"ID: {self.target_id}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def _draw_visualization(self, img, kp, conf, box):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
        for s, e in self.skeleton:
            if conf[s] > 0.5 and conf[e] > 0.5:
                cv2.line(img, (int(kp[s][0]), int(kp[s][1])), (int(kp[e][0]), int(kp[e][1])), (255, 255, 255), 2)
        for i, (x, y) in enumerate(kp):
            if conf[i] > 0.5:
                cv2.circle(img, (int(x), int(y)), 5, (255, 0, 0), -1)