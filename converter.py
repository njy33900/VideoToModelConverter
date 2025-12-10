import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import os
import torch
import datetime
import threading
import time
from typing import List, Dict, Any, Tuple

from pose_utils import *


class VideoConverter:
    """
    비디오 파일을 읽어 YOLO Pose Estimation을 수행
    LSTM 학습을 위한 시계열 데이터셋(CSV)으로 변환하는 클래스
    """

    def __init__(self, model_path: str, seq_length: int = 30):
        """
        VideoConverter를 초기화

        Args:
            model_path (str): YOLO 모델 파일 경로 (예: 'yolo11s-pose.pt')
            seq_length (int): 하나의 데이터 시퀀스로 묶을 프레임 수 (기본값: 30)
        """
        self.model_path = model_path
        self.seq_length = seq_length
        self.resize_dims = (320, 240)

        # GUI 제어
        self.stop_event = threading.Event()  # 중단 신호
        self.pause_event = threading.Event()  # 일시정지 신호

        # GPU 사용 가능 여부 확인 및 설정
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Device: {self.device}")

        self.model = YOLO(model_path)
        if self.device == '0':
            self.model.to('cuda')

        # 시각화용 뼈대 연결 정보
        self.skeleton = [
            (0, 5), (0, 6),  # 목 (코-어깨)
            (5, 7), (7, 9),  # 왼팔
            (6, 8), (8, 10),  # 오른팔
            (11, 13), (13, 15),  # 왼다리
            (12, 14), (14, 16),  # 오른다리
            (5, 6), (11, 12),  # 몸통 가로
            (5, 11), (6, 12)  # 몸통 세로
        ]

    def process_folder(self, root_dir: str, progress_callback=None) -> List[List[float]]:
        """
        지정된 루트 폴더 내의 모든 하위 폴더를 재귀적으로 순회하며 비디오를 처리

        Args:
            root_dir (str): 영상들이 저장된 최상위 폴더 경로
            progress_callback (func): (현재 파일명, 현재 진행률, 전체 진행률)을 GUI로 전달할 콜백 함수

        Returns:
            List[List[float]]: 추출된 모든 시퀀스 데이터 리스트
        """
        all_sequences = []

        # 미리보기 창 생성
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 640, 480)

        # 전체 비디오 파일 목록과 경로 생성
        video_files_to_process = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(dirpath, filename)
                    # 라벨은 부모 폴더의 이름으로 자동 지정
                    label_name = os.path.basename(dirpath)
                    video_files_to_process.append((video_path, label_name))

        total_files = len(video_files_to_process)
        if total_files == 0:
            print("처리할 영상 파일을 찾을 수 없습니다.")
            cv2.destroyAllWindows()
            return []

        print(f"총 {total_files}개의 영상 파일을 처리합니다.")

        # 파일 처리
        for idx, (video_path, label_name) in enumerate(video_files_to_process):
            # 중단 신호 체크
            if self.stop_event.is_set():
                print("\n사용자에 의해 작업이 중단되었습니다.")
                break

            current_progress = (idx + 1) / total_files * 100
            print(
                f"\n[{idx + 1}/{total_files} | {current_progress:.1f}%] 처리 중: {os.path.basename(video_path)} (Class: {label_name})")

            self._process_single_video(video_path, label_name, all_sequences, current_progress, progress_callback)

        cv2.destroyAllWindows()
        return all_sequences

    def _process_single_video(
            self,
            video_path: str,
            label: str,
            data_storage: List[List[float]],
            total_progress: float,
            progress_callback=None
    ) -> None:
        """
        단일 비디오 파일을 처리하여 시퀀스 데이터를 추출하고 저장소에 추가
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"경고: '{video_path}' 파일을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = os.path.basename(video_path)

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0: fps = 30.0

        target_interval = 0.1
        frame_step = max(1, int(fps * target_interval))

        frame_idx = 0
        temp_buffer = []
        last_valid_pose = np.zeros((17, 2))

        while not self.stop_event.is_set():
            while self.pause_event.is_set():
                if self.stop_event.is_set(): break
                time.sleep(0.1)
                cv2.waitKey(100)
            if self.stop_event.is_set(): break

            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            if frame_idx % frame_step != 0: continue

            # 레터박스를 포함한 리사이즈
            frame_resized = letterbox_resize(frame, self.resize_dims)
            results = self.model(frame_resized, verbose=False, conf=0.5, device=self.device)

            display_frame = frame_resized.copy()

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # 화면 내 가장 큰 사람을 추적
                all_boxes = results[0].boxes.xyxy.cpu().numpy()
                box_areas = (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1])
                target_index = np.argmax(box_areas)

                raw_kp = results[0].keypoints.xy.cpu().numpy()[target_index]
                conf = results[0].keypoints.conf.cpu().numpy()[target_index]
                box = results[0].boxes.xyxy.cpu().numpy()[target_index]

                filled_kp = fill_missing_keypoints(raw_kp, conf, last_valid_pose)
                last_valid_pose = filled_kp
                anchor = get_stable_anchor(filled_kp, conf)
                normalized_data = normalize_to_relative(filled_kp, anchor)
                temp_buffer.append(normalized_data)

                self._draw_visualization(display_frame, raw_kp, conf, box)

                # 데이터 저장 버퍼
                if len(temp_buffer) == self.seq_length:
                    seq = list(temp_buffer)
                    flat = np.array(seq).flatten().tolist()
                    flat.append(label)
                    flat.append(filename)
                    end_frame = frame_idx
                    start_frame = max(0, end_frame - (self.seq_length * frame_step))
                    flat.append(f"{start_frame}~{end_frame}")
                    end_sec = end_frame / fps
                    start_sec = start_frame / fps
                    time_str = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}-{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                    flat.append(time_str)
                    data_storage.append(flat)
                    cv2.circle(display_frame, (self.resize_dims[0] - 20, 20), 8, (0, 0, 255), -1)
                    temp_buffer = []

            file_progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            if progress_callback:
                progress_callback(os.path.basename(video_path), file_progress, total_progress)

            # 프리뷰(변환 진행화면) 오버레이
            cv2.putText(display_frame, f"File: {filename}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(display_frame, f"File Progress: {file_progress:.1f}%", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 255), 1)
            cv2.putText(display_frame, f"Total Progress: {total_progress:.1f}%", (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0), 1)
            cv2.putText(display_frame, f"Class: {label.upper()}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0),
                        1)

            cv2.imshow("Preview", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cap.release()

    def _draw_visualization(
            self,
            img: np.ndarray,
            kp: np.ndarray,
            conf: np.ndarray,
            box: np.ndarray
    ) -> None:
        """
        프레임 위에 객체 박스, 뼈대, 관절을 표시
        """
        # 객체 박스
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # 와이어프레임
        for s, e in self.skeleton:
            if conf[s] > 0.5 and conf[e] > 0.5 and kp[s][0] > 0 and kp[e][0] > 0:
                cv2.line(img,
                         (int(kp[s][0]), int(kp[s][1])),
                         (int(kp[e][0]), int(kp[e][1])),
                         (255, 255, 255), 1)

        # 관절 점
        for i, (x, y) in enumerate(kp):
            if conf[i] > 0.5 and x > 0:
                if i <= 4:
                    color = (255, 255, 0)
                elif i <= 10:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                cv2.circle(img, (int(x), int(y)), 3, color, -1)