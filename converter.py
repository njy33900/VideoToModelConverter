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

from pose_utils import fill_missing_keypoints, get_stable_anchor, normalize_to_relative


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

    def process_folder(self, root_dir: str, class_map: Dict[str, int], progress_callback=None) -> List[List[float]]:
        """
        지정된 루트 폴더 내의 하위 폴더들을 순회하며 비디오를 처리

        Args:
            root_dir (str): 비디오 파일들이 저장된 최상위 폴더 경로
            class_map (Dict[str, int]): 폴더명과 라벨 인덱스의 매핑 정보
            progress_callback (func): (현재수, 전체수, 파일명)을 GUI로 전달할 콜백 함수

        Returns:
            List[List[float]]: 추출된 모든 시퀀스 데이터 리스트
        """
        all_sequences = []

        # 미리보기 창 생성
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Preview", 640, 480)

        # 전체 파일 수 계산 (진행률 표시용)
        total_files = 0
        for folder_name in class_map.keys():
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.exists(folder_path):
                total_files += len(
                    [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])

        processed_count = 0

        for folder_name, label_idx in class_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.exists(folder_path):
                continue

            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"\n📂 Class [{folder_name}] - {len(files)} files")

            for idx, file in enumerate(files):
                # 중단 신호 체크
                if self.stop_event.is_set():
                    cv2.destroyAllWindows()
                    return all_sequences

                video_path = os.path.join(folder_path, file)
                self._process_single_video(video_path, label_idx, folder_name, all_sequences)

                # 진행 상황 업데이트
                processed_count += 1
                if progress_callback:
                    progress_callback(processed_count, total_files, file)

                print(f"  [{idx + 1}/{len(files)}] Done: {file}", end="\r")

        cv2.destroyAllWindows()
        return all_sequences

    def _process_single_video(
            self,
            video_path: str,
            label: int,
            class_name: str,
            data_storage: List[List[float]]
    ) -> None:
        """
        단일 비디오 파일을 처리하여 시퀀스 데이터를 추출하고 저장소에 추가
        """
        cap = cv2.VideoCapture(video_path)

        # 영상 메타데이터
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = os.path.basename(video_path)

        # 원본 해상도 가져오기
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0: fps = 30

        # FPS 기반 샘플링: 0.1초(10Hz) 간격으로 프레임 추출
        target_interval = 0.1
        frame_step = max(1, int(fps * target_interval))

        frame_idx = 0
        temp_buffer = []
        last_valid_pose = np.zeros((17, 2))

        while True:
            # 일시정지 루프
            while self.pause_event.is_set():
                if self.stop_event.is_set(): break
                time.sleep(0.1)
                cv2.waitKey(100)  # 창 응답 대기

            # 중단 신호
            if self.stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % frame_step != 0:
                continue

            # 리사이즈 & 추론
            frame = cv2.resize(frame, self.resize_dims)
            results = self.model(frame, verbose=False, conf=0.5, device=self.device)

            display_frame = frame.copy()

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                # 데이터 추출 (첫 번째 사람 기준)
                raw_kp = results[0].keypoints.xy.cpu().numpy()[0]
                conf = results[0].keypoints.conf.cpu().numpy()[0]
                box = results[0].boxes.xyxy.cpu().numpy()[0]

                # 보정 알고리즘 (pose_utils)
                filled_kp = fill_missing_keypoints(raw_kp, conf, last_valid_pose)
                last_valid_pose = filled_kp

                anchor = get_stable_anchor(filled_kp, conf)

                normalized_data = normalize_to_relative(filled_kp, anchor)
                temp_buffer.append(normalized_data)

                # 시각화 그리기
                self._draw_visualization(display_frame, raw_kp, conf, box)

                # 버퍼가 설정된 길이만큼 차면 시퀀스 저장
                if len(temp_buffer) == self.seq_length:
                    seq = list(temp_buffer)
                    flat = np.array(seq).flatten().tolist()
                    flat.append(label)

                    # 메타데이터 추가
                    flat.append(filename)  # 파일명

                    # 프레임 구간
                    end_frame = frame_idx
                    start_frame = max(0, end_frame - (self.seq_length * frame_step))
                    flat.append(f"{start_frame}~{end_frame}")

                    # 시간 정보 (MM:SS)
                    end_sec = end_frame / fps
                    start_sec = start_frame / fps
                    time_str = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}-{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                    flat.append(time_str)

                    data_storage.append(flat)

                    # 저장 완료 표시
                    cv2.circle(display_frame, (self.resize_dims[0] - 20, 20), 8, (0, 0, 255), -1)
                    temp_buffer = []

            # 정보 오버레이
            cv2.putText(display_frame, f"File: {filename}", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            cv2.putText(display_frame, f"Frame: {frame_idx}/{total_frames} ({progress:.1f}%)", (5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

            class_color = (255, 255, 255)
            if label == 0:
                class_color = (0, 255, 0)
            elif label == 1:
                class_color = (0, 255, 255)
            elif label == 2:
                class_color = (0, 0, 255)

            cv2.putText(display_frame, f"Class: {class_name.upper()}", (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, class_color, 1)

            resize_text = f"Res: {orig_w}x{orig_h} > {self.resize_dims[0]}x{self.resize_dims[1]}"
            cv2.putText(display_frame, resize_text, (5, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

            cv2.imshow("Preview", display_frame)

            # 키보드 Q로 중단
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                cap.release()
                cv2.destroyAllWindows()
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