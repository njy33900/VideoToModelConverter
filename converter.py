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
from collections import deque

from pose_utils import *

class VideoConverter:
    """
    비디오 파일을 읽어 YOLO Pose Estimation을 수행 (Tracking 적용)
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
        self.resize_dims = (640, 640)

        # FPS 동기화를 위한 타겟 FPS 설정
        self.target_fps = 10

        # 시각화 옵션 (속도 최적화를 위해 False 권장)
        self.vis_enabled = True

        # GUI 제어
        self.stop_event = threading.Event()  # 중단 신호
        self.pause_event = threading.Event()  # 일시정지 신호

        # GPU 사용 가능 여부 확인 및 설정
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Device: {self.device}")

        # 추적(Tracking) 지원 모델 로드
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
        # 메모리 관리: 대량 데이터 처리 시 여기서 바로 CSV로 저장하고
        # 리스트를 비우는 방식(Generator)을 권장하지만, 기존 구조 호환을 위해 리스트 유지.
        all_sequences = []

        # 미리보기 창 생성
        if self.vis_enabled:
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
            if self.vis_enabled: cv2.destroyAllWindows()
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

        if self.vis_enabled:
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
        Tracking 적용 + Scale Normalization + Jittering Filter + FPS Sync
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"경고: '{video_path}' 파일을 열 수 없습니다.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        filename = os.path.basename(video_path)
        if fps == 0: fps = 30.0

        # 타겟 FPS에 맞춘 프레임 스텝 계산 (시간 동기화)
        frame_step = max(1, int(fps / self.target_fps))

        frame_idx = 0

        # 데이터 버퍼
        position_buffer = deque(maxlen=self.seq_length)

        # 지터링 보정을 위한 이동 평균(Moving Average) 버퍼
        raw_kps_buffer = deque(maxlen=3)  # 최근 3프레임 평균

        # 초기값 처리: 첫 유효 프레임 전까지 None 유지
        last_valid_pose = None

        # 객체 추적을 위한 ID 변수
        target_id = None

        data_generated_for_this_video = False
        SLIDING_STRIDE = 2
        frames_since_last_save = SLIDING_STRIDE
        MIN_SAMPLES_FOR_RESAMPLING = int(self.seq_length * 0.7)  # 약 70% 이상이면 리샘플링

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

            # 리사이즈
            frame_resized = letterbox_resize(frame, self.resize_dims)
            display_frame = frame_resized.copy()

            # YOLO Tracking 수행 (persist=True로 ID 유지)
            # tracker는 botsort.yaml 또는 bytetrack.yaml 사용
            results = self.model.track(frame_resized, persist=True, verbose=False, conf=0.5, device=self.device,
                                       tracker="bytetrack.yaml")

            current_kp = None
            current_conf = None
            current_box = None

            # 감지된 객체가 있는지 확인
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.xy.cpu().numpy()
                confs = results[0].keypoints.conf.cpu().numpy()

                # 타겟 ID 선정 로직
                if target_id is None:
                    # 초기 타겟: 박스 면적이 가장 큰 사람을 메인으로 선정
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    if len(areas) > 0:
                        max_idx = np.argmax(areas)
                        target_id = track_ids[max_idx]

                # 타겟 데이터 추출
                if target_id in track_ids:
                    idx = track_ids.index(target_id)
                    current_kp = keypoints[idx]
                    current_conf = confs[idx]
                    current_box = boxes[idx]

            # 데이터 가공
            norm_kp = np.zeros(34)
            valid_frame_processed = False

            if current_kp is not None:
                # 초기값 초기화 (최초 1회)
                if last_valid_pose is None:
                    last_valid_pose = current_kp

                # 결측치 보간 (기존 로직 활용)
                filled_kp = fill_missing_keypoints(current_kp, current_conf, last_valid_pose)
                last_valid_pose = filled_kp

                # 지터링 보정 (Moving Average)
                raw_kps_buffer.append(filled_kp)
                smoothed_kp = np.mean(np.array(raw_kps_buffer), axis=0)

                # 중심점(Anchor) 계산
                anchor = get_stable_anchor(smoothed_kp, current_conf)  # smooth된 값 사용

                # anchor가 None일 경우 예외 처리 (박스 중심으로 대체)
                if anchor is None:
                    # 박스의 중심점 (Center of Bounding Box) 계산
                    cx = (current_box[0] + current_box[2]) / 2
                    cy = (current_box[1] + current_box[3]) / 2
                    anchor = np.array([cx, cy])

                # 스케일 정규화 (Scale Normalization)
                # 박스 높이 또는 몸통 길이를 기준으로 정규화하여 거리 불변성 확보
                box_h = current_box[3] - current_box[1]
                scale_factor = max(box_h, 1.0)  # 0 나누기 방지

                # 상대 좌표 계산 후 스케일로 나누기
                # pose_utils의 normalize_to_relative가 단순히 (kp-anchor)라면, 여기서 직접 나눔
                relative_kp = smoothed_kp - anchor
                # 0~1 사이로 정규화하기 위해 (Scale Normalization)
                norm_kp_vec = relative_kp / scale_factor
                norm_kp = norm_kp_vec.flatten()  # 1차원 배열로 변환

                valid_frame_processed = True

                if self.vis_enabled:
                    self._draw_visualization(display_frame, smoothed_kp, current_conf, current_box, target_id)

            # 버퍼 관리
            # 데이터 품질을 위해 타겟을 놓치면(valid_frame_processed=False) 버퍼에 추가하지 않거나
            # 짧은 결측은 이전 데이터로 채우는 등의 전략이 필요. 여기서는 타겟 감지시에만 추가.
            if valid_frame_processed:
                position_buffer.append(norm_kp)
                frames_since_last_save += 1

                # 시퀀스 데이터 생성
                if len(position_buffer) == self.seq_length:
                    if frames_since_last_save >= SLIDING_STRIDE:
                        pos_seq = np.array(position_buffer)
                        vel_seq = np.diff(pos_seq, axis=0)
                        acc_seq = np.diff(vel_seq, axis=0)

                        vel_seq_padded = np.pad(vel_seq, ((1, 0), (0, 0)), 'constant')
                        acc_seq_padded = np.pad(acc_seq, ((2, 0), (0, 0)), 'constant')

                        full_feature_seq = np.concatenate((pos_seq, vel_seq_padded, acc_seq_padded), axis=1)
                        flat = full_feature_seq.flatten().tolist()

                        flat.append(label)
                        flat.append(filename)

                        end_frame = frame_idx
                        start_frame = max(0, end_frame - (self.seq_length * frame_step))
                        flat.append(f"{start_frame}~{end_frame}")

                        end_sec, start_sec = end_frame / fps, start_frame / fps
                        time_str = f"{int(start_sec // 60):02d}:{int(start_sec % 60):02d}-{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                        flat.append(time_str)
                        data_storage.append(flat)

                        if self.vis_enabled:
                            cv2.circle(display_frame, (self.resize_dims[0] - 20, 20), 8, (0, 0, 255), -1)

                        data_generated_for_this_video = True
                        frames_since_last_save = 0

            # 진행률 표시
            file_progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            if progress_callback:
                progress_callback(os.path.basename(video_path), file_progress, total_progress)

            if self.vis_enabled:
                cv2.putText(display_frame, f"File: {filename}", (5, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Target ID: {target_id}", (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(display_frame, f"Buffer: {len(position_buffer)}/{self.seq_length}", (5, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                cv2.imshow("Preview", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break

        # 영상 처리 완료 후, 하이브리드 로직 (리샘플링)
        # 데이터가 하나도 생성되지 않았고, 버퍼에 데이터가 어느 정도 있다면 보간
        if not data_generated_for_this_video and len(position_buffer) > 0:

            if len(position_buffer) >= MIN_SAMPLES_FOR_RESAMPLING:
                print(f"  -> 영상이 짧아 리샘플링을 적용합니다. ({len(position_buffer)} -> {self.seq_length} 프레임)")

                original_sequence = np.array(position_buffer)
                original_len = len(original_sequence)
                target_len = self.seq_length

                original_x = np.linspace(0, 1, original_len)
                target_x = np.linspace(0, 1, target_len)

                resampled_sequence = np.zeros((target_len, 34))
                for i in range(34):
                    resampled_sequence[:, i] = np.interp(target_x, original_x, original_sequence[:, i])

                pos_seq = resampled_sequence
                vel_seq = np.diff(pos_seq, axis=0)
                acc_seq = np.diff(vel_seq, axis=0)
                vel_seq_padded = np.pad(vel_seq, ((1, 0), (0, 0)), 'constant')
                acc_seq_padded = np.pad(acc_seq, ((2, 0), (0, 0)), 'constant')
                full_feature_seq = np.concatenate((pos_seq, vel_seq_padded, acc_seq_padded), axis=1)
                flat = full_feature_seq.flatten().tolist()

                flat.append(label)
                flat.append(filename)
                flat.append(f"0~{total_frames} (Resampled)")
                end_sec = total_frames / fps
                time_str = f"00:00-{int(end_sec // 60):02d}:{int(end_sec % 60):02d}"
                flat.append(time_str)
                data_storage.append(flat)

            else:
                print(f"  -> 영상이 너무 짧거나 타겟을 놓쳤습니다. ({len(position_buffer)} 샘플) 데이터 생성 제외.")

        cap.release()

    def _draw_visualization(
            self,
            img: np.ndarray,
            kp: np.ndarray,
            conf: np.ndarray,
            box: np.ndarray,
            track_id: int
    ) -> None:
        """
        프레임 위에 객체 박스, 뼈대, 관절을 표시 (Target ID 포함)
        """
        # 객체 박스
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # ID 표시
        cv2.putText(img, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

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