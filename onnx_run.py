import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from collections import deque
import torch
import os
import json
import time
import tkinter as tk
from tkinter import filedialog

# 불필요한 로그 제거
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pose_utils import fill_missing_keypoints, get_stable_anchor, letterbox_resize


class OnnxVideoTester:
    """
    ONNX 모델(FP16)을 사용하여 실시간 행동 감지를 수행하는 클래스
    """

    def __init__(self, yolo_model_path: str, onnx_model_path: str):
        self.seq_length = 30
        self.resize_dims = (640, 640)

        # 시간 동기화 (10 FPS)
        self.target_fps = 10
        self.interval = 1.0 / self.target_fps
        self.last_sampling_time = 0

        self.buffer = deque(maxlen=self.seq_length)
        self.last_valid_pose = None
        self.target_id = None

        # ONNX Runtime 세션 초기화
        print(f"🚀 Loading ONNX model: {onnx_model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
        except Exception as e:
            print(f"⚠️ ONNX 로드 실패: {e}")
            raise

        # 입력 노드 이름 확인
        self.input_name = self.ort_session.get_inputs()[0].name

        # YOLO 모델 로딩
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Device: {self.device}")
        self.yolo_model = YOLO(yolo_model_path)
        if self.device == '0':
            self.yolo_model.to('cuda')

        # 클래스 네임 로드
        self.class_names = []
        base_path = onnx_model_path.replace('_fp16.onnx', '').replace('.onnx', '')

        # 우선순위
        possible_paths = [base_path + "_fp16_classes.json", base_path + "_classes.json"]

        class_loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.class_names = json.load(f)
                print(f"✅ 클래스 파일 로드됨: {path}")
                class_loaded = True
                break

        if not class_loaded:
            print(f"⚠️ 클래스 맵 파일을 찾을 수 없습니다. (예상 경로: {possible_paths})")

        self.current_prediction = "System Ready. Gathering Data..."
        self.skeleton = [
            (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (5, 6), (11, 12), (5, 11), (6, 12)
        ]

    def process_frame(self, frame: np.ndarray) -> (str, np.ndarray):
        current_time = time.time()
        frame_resized = letterbox_resize(frame, self.resize_dims)
        display_frame = frame_resized.copy()

        # YOLO Tracking
        results = self.yolo_model.track(frame_resized, persist=True, verbose=False, conf=0.5,
                                        device=self.device, tracker="bytetrack.yaml")

        current_kp, current_conf, current_box = None, None, None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.xy.cpu().numpy()
            confs = results[0].keypoints.conf.cpu().numpy()

            # 타겟 ID 선정
            if self.target_id is None:
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                self.target_id = track_ids[np.argmax(areas)]

            if self.target_id in track_ids:
                idx = track_ids.index(self.target_id)
                current_kp, current_conf, current_box = keypoints[idx], confs[idx], boxes[idx]
                self._draw_visualization(display_frame, current_kp, current_conf, current_box)
            else:
                self.target_id = None

        # 10 FPS 샘플링 로직
        if current_time - self.last_sampling_time >= self.interval:
            self.last_sampling_time = current_time
            norm_kp = np.zeros(34)

            if current_kp is not None:
                if self.last_valid_pose is None: self.last_valid_pose = current_kp
                filled_kp = fill_missing_keypoints(current_kp, current_conf, self.last_valid_pose)
                self.last_valid_pose = filled_kp

                anchor = get_stable_anchor(filled_kp, current_conf)
                if anchor is None:
                    anchor = np.array([(current_box[0] + current_box[2]) / 2, (current_box[1] + current_box[3]) / 2])

                box_h = current_box[3] - current_box[1]
                norm_kp = ((filled_kp - anchor) / max(box_h, 1.0)).flatten()

            self.buffer.append(norm_kp)

            if len(self.buffer) < self.seq_length:
                self.current_prediction = f"Buffering... {len(self.buffer)}/{self.seq_length}"
            else:
                self._run_onnx_prediction()

        self._draw_overlay(display_frame)
        return self.current_prediction, display_frame

    def _run_onnx_prediction(self):
        # 1. 특징 추출 (Pos, Vel, Acc)
        pos_seq = np.array(self.buffer)
        vel_seq = np.pad(np.diff(pos_seq, axis=0), ((1, 0), (0, 0)), 'constant')
        acc_seq = np.pad(np.diff(vel_seq, axis=0), ((1, 0), (0, 0)), 'constant')

        full_feature_seq = np.concatenate((pos_seq, vel_seq, acc_seq), axis=1)  # (30, 102)

        # FP16 모델이므로 입력 데이터도 float16으로 변환해야 함
        input_data = np.expand_dims(full_feature_seq, axis=0).astype(np.float16)

        # ONNX 추론 실행
        outputs = self.ort_session.run(None, {self.input_name: input_data})
        prediction_probs = outputs[0][0]

        # 결과 해석
        top_3_indices = np.argsort(prediction_probs)[::-1][:3]
        results = []
        for i in top_3_indices:
            name = self.class_names[i] if i < len(self.class_names) else f"ID_{i}"
            results.append(f"{name.upper()}: {prediction_probs[i]:.0%}")

        self.current_prediction = "\n".join(results)

    def _draw_overlay(self, img):
        lines = self.current_prediction.split('\n')
        # 상태에 따라 색상 변경 (Buffering: 노랑, 완료: 초록)
        color = (0, 255, 255) if "Buffering" in lines[0] else (0, 255, 0)

        # 여러 줄 출력 지원
        for i, line in enumerate(lines):
            y_pos = 30 + (i * 30)
            cv2.putText(img, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _draw_visualization(self, img, kp, conf, box):
        x1, y1, x2, y2 = map(int, box)

        # [수정 1] 박스 색상: 초록(0, 255, 0), 두께: 얇게(1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 스켈레톤 라인 (기존 유지 - 흰색)
        for s, e in self.skeleton:
            if conf[s] > 0.5 and conf[e] > 0.5:
                cv2.line(img, (int(kp[s][0]), int(kp[s][1])), (int(kp[e][0]), int(kp[e][1])), (255, 255, 255), 1)

        # [수정 2] 관절 포인트 색상 분리 (머리/상체/하체)
        for i, (x, y) in enumerate(kp):
            if conf[i] > 0.5:
                # 색상 지정 (BGR 형식)
                if i <= 4:  # 머리 (코, 눈, 귀) -> 노랑
                    color = (0, 255, 255)
                elif i <= 10:  # 상체 (어깨, 팔, 손목) -> 자홍(Magenta)
                    color = (255, 0, 255)
                else:  # 하체 (골반, 다리, 발목) -> 파랑
                    color = (255, 0, 0)

                # 포인트 그리기
                cv2.circle(img, (int(x), int(y)), 4, color, -1)


# ==========================================
# 실행부: 파일 브라우저 통합
# ==========================================
def select_video_file():
    """Tkinter를 사용하여 파일 선택 창을 띄웁니다."""
    root = tk.Tk()
    root.withdraw()  # 빈 부모 창 숨기기

    file_path = filedialog.askopenfilename(
        title="분석할 영상 파일을 선택하세요 (MP4, AVI, MKV)",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov"), ("All Files", "*.*")]
    )

    root.destroy()
    return file_path


if __name__ == "__main__":
    # 변환된 FP16 ONNX 모델 경로
    # ONNX_MODEL_PATH = "action_model_fp16.onnx"
    ONNX_MODEL_PATH = "action_model_v0.95_fp16.onnx"
    # YOLO 모델 경로
    YOLO_MODEL_PATH = "yolo11n-pose.pt"

    # 영상 파일 선택
    print("📂 파일 탐색기를 여는 중...")
    video_path = select_video_file()

    if not video_path:
        print("❌ 영상이 선택되지 않았습니다. 종료합니다.")
    else:
        print(f"🎬 선택된 영상: {video_path}")

        # 모델 파일 존재 확인
        if not os.path.exists(ONNX_MODEL_PATH):
            print(f"❌ ONNX 모델을 찾을 수 없습니다: {ONNX_MODEL_PATH}")
            print("   onnx_convert.py를 먼저 실행하여 모델을 변환해주세요.")
        elif not os.path.exists(YOLO_MODEL_PATH):
            print(f"❌ YOLO 모델을 찾을 수 없습니다: {YOLO_MODEL_PATH}")
        else:
            try:
                # 테스터 초기화
                tester = OnnxVideoTester(
                    yolo_model_path=YOLO_MODEL_PATH,
                    onnx_model_path=ONNX_MODEL_PATH
                )

                # 영상 처리 시작
                cap = cv2.VideoCapture(video_path)

                print("▶️ 재생 시작 (종료하려면 화면을 클릭하고 'q'를 누르세요)")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("🏁 영상 종료")
                        break

                    # 행동 감지 수행
                    prediction, result_frame = tester.process_frame(frame)

                    # 화면 출력
                    cv2.imshow("High-Performance Action Detection (ONNX FP16)", result_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()

            except Exception as e:
                print(f"❌ 실행 중 오류 발생: {e}")
                import traceback

                traceback.print_exc()