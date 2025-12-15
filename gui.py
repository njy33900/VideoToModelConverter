import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import os
import glob
import threading
import sys
import pandas as pd
import datetime
import cv2
import time
from PIL import Image, ImageTk
from collections import deque
import traceback

# 모듈 임포트
from converter import VideoConverter
from trainer.train_logic import ModelTrainer
from trainer.train_logic_transformer import TransformerTrainer
from tester import VideoTester
import tensorflow as tf


# 콘솔 리다이렉션 클래스
class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        try:
            self.widget.after(0, self._append, str)
        except:
            pass

    def _append(self, str):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, str, (self.tag,))
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass


class MainGUI:
    def __init__(self, root, video_root, save_root, class_map):
        self.root = root
        self.video_root = video_root
        self.save_root = save_root
        self.class_map = class_map

        self.root.title("행동 데이터 변환 및 학습기")
        self.root.geometry("1100x800")

        # 인스턴스
        self.converter = None
        # self.trainer = ModelTrainer()
        self.current_trainer = None
        self.tester = None

        self.is_converting = False
        self.is_training = False
        self.is_testing = False

        self.preview_thread = None
        self.stop_preview_event = threading.Event()

        # 테스트 스레드 제어용 이벤트
        self.test_thread = None
        self.stop_test_event = threading.Event()

        self.source_data_path = tk.StringVar(value="폴더를 선택해주세요.")

        self._init_layout()

        # 로그 리다이렉션
        self.redirector = TextRedirector(self.log_text)
        self.original_stdout = sys.stdout
        sys.stdout = self.redirector

        # 초기 로딩
        self._load_models_combo()
        # self._load_csv_list()
        self._load_h5_models_combo()

        # 창 닫기 이벤트 바인딩
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_layout(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        self.tab_converter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_converter, text=" 데이터 변환 (Video -> CSV) ")
        self.tab_trainer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trainer, text=" 모델 학습 (CSV -> AI Model) ")
        self.tab_tester = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tester, text=" 동작 테스트 ")

        log_frame = ttk.LabelFrame(self.root, text="시스템 로그")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text = ScrolledText(log_frame, state='disabled', height=10, bg="black", fg="#00ff00",
                                     font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self._init_converter_tab(self.tab_converter)
        self._init_trainer_tab(self.tab_trainer)
        self._init_tester_tab(self.tab_tester)

    def on_tab_changed(self, event):
        if self.is_testing:
            self.stop_testing()

    def _init_converter_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.LabelFrame(paned, text="인식된 폴더 및 파일 목록 (더블클릭하여 미리보기)")
        paned.add(left_frame, weight=1)
        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_scan_results = ttk.Treeview(left_frame, yscrollcommand=scrollbar.set)
        self.tree_scan_results.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.tree_scan_results.yview)
        self.tree_scan_results.bind("<Double-1>", self.on_scan_tree_double_click)

        right_frame = tk.Frame(paned)
        paned.add(right_frame, weight=1)
        source_frame = ttk.LabelFrame(right_frame, text="원본 영상 폴더 선택")
        source_frame.pack(fill=tk.X, padx=10, pady=10)
        lbl_path = tk.Label(source_frame, textvariable=self.source_data_path, anchor='w', justify='left')
        lbl_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        btn_browse = ttk.Button(source_frame, text="폴더 찾아보기", command=self._browse_source_folder)
        btn_browse.pack(side=tk.RIGHT, padx=5, pady=5)
        model_frame = ttk.LabelFrame(right_frame, text="YOLO 모델 선택")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        self.combo_models = ttk.Combobox(model_frame, state="readonly")
        self.combo_models.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        btn_refresh = ttk.Button(model_frame, text="새로고침", command=self._load_models_combo)
        btn_refresh.pack(side=tk.RIGHT, padx=5, pady=5)
        ctrl_frame = ttk.LabelFrame(right_frame, text="변환 실행 및 제어")
        ctrl_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        progress_panel = tk.Frame(ctrl_frame)
        progress_panel.pack(fill=tk.X, pady=10)
        self.lbl_conv_status = tk.Label(progress_panel, text="대기 중", anchor='w')
        self.lbl_conv_status.pack(fill=tk.X)
        tk.Label(progress_panel, text="파일 진행률:").pack(side=tk.LEFT, padx=5)
        self.file_progress = tk.DoubleVar()
        ttk.Progressbar(progress_panel, variable=self.file_progress, maximum=100).pack(side=tk.LEFT, fill=tk.X,
                                                                                       expand=True)
        tk.Label(progress_panel, text="전체 진행률:").pack(side=tk.LEFT, padx=5)
        self.total_progress = tk.DoubleVar()
        ttk.Progressbar(progress_panel, variable=self.total_progress, maximum=100).pack(side=tk.LEFT, fill=tk.X,
                                                                                        expand=True)
        btn_box = tk.Frame(ctrl_frame)
        btn_box.pack(pady=20)
        self.btn_start = tk.Button(btn_box, text="▶ 변환 시작", bg="#ccffcc", height=2, width=15,
                                   command=self.start_conversion)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        self.btn_pause = tk.Button(btn_box, text="⏸ 일시정지", bg="#ffeb99", height=2, width=15, state="disabled",
                                   command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=10)
        self.btn_stop = tk.Button(btn_box, text="⏹ 중단", bg="#ffcccc", height=2, width=15, state="disabled",
                                  command=self.stop_conversion)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

    def _init_trainer_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 좌측 창: CSV 목록
        left_frame = ttk.LabelFrame(paned, text="📊 학습 데이터셋(CSV) 목록")
        paned.add(left_frame, weight=1)

        # 폴더 선택 UI
        folder_frame = tk.Frame(left_frame)
        folder_frame.pack(fill=tk.X, padx=5, pady=5)
        self.csv_source_path = tk.StringVar(value="CSV 폴더를 선택해주세요.")
        lbl_csv_path = tk.Label(folder_frame, textvariable=self.csv_source_path, anchor='w')
        lbl_csv_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        btn_browse_csv = ttk.Button(folder_frame, text="폴더 선택", command=self._browse_csv_folder)
        btn_browse_csv.pack(side=tk.RIGHT)

        # CSV 파일 목록 Treeview
        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree_csv_scan = ttk.Treeview(left_frame, columns=("filename", "size"), show="headings", yscrollcommand=scrollbar.set)
        self.tree_csv_scan.heading("filename", text="파일명")
        self.tree_csv_scan.heading("size", text="크기")
        self.tree_csv_scan.column("filename", width=200)
        self.tree_csv_scan.column("size", width=80, anchor='center')
        self.tree_csv_scan.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.tree_csv_scan.yview)

        # 우측 창: 학습 제어
        right_frame = tk.Frame(paned)
        paned.add(right_frame, weight=1)

        setting_frame = ttk.LabelFrame(right_frame, text="학습 파라미터")
        setting_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(setting_frame, text="학습 엔진:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.combo_trainer_engine = ttk.Combobox(setting_frame, state="readonly",
                                                 values=['LSTM', 'Transformer'])
        self.combo_trainer_engine.current(0)
        self.combo_trainer_engine.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Label(setting_frame, text="Epochs (반복 횟수):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ent_epochs = ttk.Entry(setting_frame)
        self.ent_epochs.insert(0, "100") # Epoch 기본값 증가
        self.ent_epochs.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(setting_frame, text="Batch Size (배치 크기):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.ent_batch = ttk.Entry(setting_frame)
        self.ent_batch.insert(0, "32")
        self.ent_batch.grid(row=2, column=1, padx=5, pady=5)

        ctrl_frame = ttk.LabelFrame(right_frame, text="학습 제어")
        ctrl_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        self.train_progress = tk.DoubleVar()
        ttk.Progressbar(ctrl_frame, variable=self.train_progress, maximum=100).pack(fill=tk.X, padx=10, pady=20)
        self.lbl_train_status = tk.Label(ctrl_frame, text="CSV 폴더를 선택하고 학습을 시작하세요.", fg="gray", font=("Arial", 11))
        self.lbl_train_status.pack(pady=10)
        self.btn_train_start = tk.Button(ctrl_frame, text="🚀 학습 시작", bg="#ccffcc", font=("Arial", 14, "bold"), height=2,
                                         command=self.start_training)
        self.btn_train_start.pack(fill=tk.X, padx=20, pady=10)

    def _browse_csv_folder(self):
        if self.is_training: return
        folder_selected = filedialog.askdirectory(title="학습할 CSV 파일이 담긴 폴더를 선택하세요")
        if folder_selected:
            self.csv_source_path.set(folder_selected)
            # 기존 목록 지우기
            for item in self.tree_csv_scan.get_children():
                self.tree_csv_scan.delete(item)
            # CSV 파일 스캔하여 Treeview에 추가
            csv_files = glob.glob(os.path.join(folder_selected, "*.csv"))
            for f_path in sorted(csv_files):
                fname = os.path.basename(f_path)
                size = f"{os.path.getsize(f_path) / (1024 * 1024):.1f} MB"
                self.tree_csv_scan.insert("", "end", values=(fname, size))

    def _init_tester_tab(self, parent):
        # PanedWindow를 사용하여 좌/우 분할 레이아웃 생성
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_container = tk.Frame(paned, width=300)
        paned.add(left_container, weight=1)


        # 모델 선택
        model_frame = ttk.LabelFrame(left_container, text="모델 선택")
        model_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(model_frame, text="YOLO (.pt):").pack(anchor='w', padx=5)
        self.combo_yolo_test = ttk.Combobox(model_frame, state="readonly")
        self.combo_yolo_test.pack(fill=tk.X, padx=5, pady=2)

        tk.Label(model_frame, text="LSTM (.h5):").pack(anchor='w', padx=5, pady=(10, 0))
        self.combo_lstm_test = ttk.Combobox(model_frame, state="readonly")
        self.combo_lstm_test.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(model_frame, text="모델 목록 새로고침", command=self._load_h5_models_combo).pack(pady=5)

        # 영상 선택
        video_frame = ttk.LabelFrame(left_container, text="영상 선택")
        video_frame.pack(fill=tk.X, padx=10, pady=10)

        self.video_path_test = tk.StringVar(value="영상을 불러오세요.")
        tk.Label(video_frame, textvariable=self.video_path_test, wraplength=280).pack(padx=5, pady=5)
        ttk.Button(video_frame, text="영상 불러오기", command=self._browse_video).pack(fill=tk.X, padx=5, pady=5)

        # 예측 결과
        result_frame = ttk.LabelFrame(left_container, text="실시간 예측 결과")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.lbl_prediction = tk.Label(result_frame, text="대기 중", font=("Arial", 16, "bold"), fg="gray", justify='left')
        self.lbl_prediction.pack(expand=True)

        # 제어 버튼
        ctrl_frame = ttk.LabelFrame(left_container, text="실행 제어")
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)

        btn_box = tk.Frame(ctrl_frame)
        btn_box.pack(pady=5, fill=tk.X, expand=True)

        self.btn_test_start = tk.Button(btn_box, text="▶ 테스트 시작", bg="#ccffcc", height=2, command=self.start_testing)
        self.btn_test_start.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.btn_test_stop = tk.Button(btn_box, text="⏹ 중단", bg="#ffcccc", height=2, state="disabled",
                                       command=self.stop_testing)
        self.btn_test_stop.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # 미리보기
        right_container = ttk.LabelFrame(paned, text="미리보기")
        paned.add(right_container, weight=3)

        self.lbl_test_preview = tk.Label(right_container, bg="black")
        self.lbl_test_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _load_models_combo(self):
        pt_files = glob.glob("*.pt")
        if not pt_files: pt_files = ["모델 없음"]
        self.combo_models['values'] = pt_files
        self.combo_models.current(0)
        if hasattr(self, 'combo_yolo_test'):
            self.combo_yolo_test['values'] = pt_files
            self.combo_yolo_test.current(0)

    def _load_csv_list(self):
        for item in self.tree_csv.get_children(): self.tree_csv.delete(item)
        if not os.path.exists(self.save_root): os.makedirs(self.save_root)
        files = glob.glob(os.path.join(self.save_root, "*.csv"))
        files.sort(key=os.path.getmtime, reverse=True)
        for f in files:
            size = f"{os.path.getsize(f) / 1024:.1f} KB"
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M')
            fname = os.path.basename(f)
            self.tree_csv.insert("", "end", values=(fname, size, mtime))

    def _load_h5_models_combo(self):
        model_dir = os.path.join("trainer", "models")
        if not os.path.exists(model_dir): os.makedirs(model_dir)
        h5_files = glob.glob(os.path.join(model_dir, "*.h5"))
        h5_files = [os.path.basename(f) for f in h5_files]
        if not h5_files: h5_files = ["학습된 모델 없음"]
        if hasattr(self, 'combo_lstm_test'):
            self.combo_lstm_test['values'] = h5_files
            self.combo_lstm_test.current(0)

    def _browse_source_folder(self):
        if self.is_converting: return
        folder_selected = filedialog.askdirectory(title="데이터가 포함된 최상위 폴더를 선택하세요")
        if folder_selected:
            self.source_data_path.set(folder_selected)
            print(f"데이터 소스 폴더 선택됨: {folder_selected}")
            self._scan_and_display_folder(folder_selected)

    def _scan_and_display_folder(self, root_dir):
        for item in self.tree_scan_results.get_children():
            self.tree_scan_results.delete(item)
        for dirpath, dirnames, filenames in os.walk(root_dir):
            video_files = [f for f in filenames if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if video_files:
                relative_path = os.path.relpath(dirpath, root_dir)
                parts = relative_path.split(os.sep)
                current_parent = ""
                for part in parts:
                    node_found = False
                    for child in self.tree_scan_results.get_children(current_parent):
                        if self.tree_scan_results.item(child, "text") == f"📂 {part}":
                            current_parent = child
                            node_found = True
                            break
                    if not node_found:
                        current_parent = self.tree_scan_results.insert(current_parent, "end", text=f"📂 {part}",
                                                                       open=True)
                for file in sorted(video_files):
                    self.tree_scan_results.insert(current_parent, "end", text=file)

    def on_scan_tree_double_click(self, event):
        item_id = self.tree_scan_results.focus()
        if not item_id or self.tree_scan_results.get_children(item_id):
            return
        path_parts = [self.tree_scan_results.item(item_id, 'text')]
        parent_id = self.tree_scan_results.parent(item_id)
        while parent_id:
            parent_text = self.tree_scan_results.item(parent_id, 'text')
            folder_name = parent_text.replace('📂', '').strip()
            path_parts.insert(0, folder_name)
            parent_id = self.tree_scan_results.parent(parent_id)
        root_path = self.source_data_path.get()
        if not os.path.isdir(root_path):
            messagebox.showwarning("경고", "먼저 유효한 데이터 폴더를 선택해주세요.")
            return
        full_video_path = os.path.join(root_path, *path_parts)
        if os.path.exists(full_video_path):
            threading.Thread(target=self._popup_preview, args=(full_video_path,), daemon=True).start()
        else:
            messagebox.showerror("오류", f"파일을 찾을 수 없습니다:\n{full_video_path}")

    def _popup_preview(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"오류: '{path}' 영상을 열 수 없습니다.")
            return
        win_name = f"미리보기: {os.path.basename(path)}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 640, 480)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow(win_name, frame)
            if cv2.waitKey(33) & 0xFF == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyWindow(win_name)

    def start_conversion(self):
        model_path = self.combo_models.get()
        source_dir = self.source_data_path.get()
        if "없음" in model_path: return messagebox.showerror("오류", "YOLO 모델을 선택해주세요.")
        if not os.path.isdir(source_dir): return messagebox.showerror("오류", "유효한 데이터 폴더를 선택해주세요.")
        self.is_converting = True
        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.converter = VideoConverter(model_path=model_path, seq_length=30)
        threading.Thread(target=self._run_conversion, args=(source_dir,), daemon=True).start()

    def stop_conversion(self):
        if self.converter and messagebox.askyesno("확인", "중단하시겠습니까?"): self.converter.stop_event.set()

    def toggle_pause(self):
        if not self.converter: return
        if self.converter.pause_event.is_set():
            self.converter.pause_event.clear()
            self.btn_pause.config(text="⏸ 일시정지", bg="#ffeb99")
            self.lbl_conv_status.config(text="재개됨...")
        else:
            self.converter.pause_event.set()
            self.btn_pause.config(text="▶ 다시 시작", bg="#ccffcc")
            self.lbl_conv_status.config(text="일시정지 중...")

    def _run_conversion(self, source_dir: str):
        try:
            def progress_update_callback(filename, file_pct, total_pct):
                self.root.after(0, self.lbl_conv_status.config, {'text': f"처리 중: {filename}"})
                self.file_progress.set(file_pct)
                self.total_progress.set(total_pct)

            dataset = self.converter.process_folder(source_dir, progress_callback=progress_update_callback)
            self.root.after(0, self._finish_conversion, dataset)
        except Exception as e:
            error_message = traceback.format_exc()
            self.root.after(0, lambda msg=error_message: messagebox.showerror("변환 오류", msg))
        finally:
            self.root.after(0, self._reset_conv_ui)

    def _finish_conversion(self, dataset):
        if dataset:
            chunk_size_mb = 200
            bytes_per_row = 30 * 102 * 10
            rows_per_chunk = int((chunk_size_mb * 1024 * 1024) / bytes_per_row)

            num_rows = len(dataset)
            # 데이터가 없을 경우 처리
            if num_rows == 0:
                messagebox.showinfo("정보", "변환된 데이터가 없습니다.")
                self._reset_conv_ui()
                return

            num_chunks = (num_rows // rows_per_chunk) + (1 if num_rows % rows_per_chunk > 0 else 0)

            # 타임스탬프 기반으로 새로운 하위 폴더 경로 생성
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            folder_name = f"pose_data_{ts}"
            save_folder_path = os.path.join(self.save_root, folder_name)

            # 실제로 하위 폴더를 생성
            os.makedirs(save_folder_path, exist_ok=True)

            # 컬럼 헤더 정의
            cols = [f'v{i}' for i in range(self.converter.seq_length * 102)] + ['label', 'filename', 'frame', 'time']

            saved_files_info = []
            for i in range(num_chunks):
                start_idx = i * rows_per_chunk
                end_idx = start_idx + rows_per_chunk
                chunk_data = dataset[start_idx:end_idx]

                # 분할된 파일 이름은 단순하게 part_N.csv로 지정
                chunk_filename = f"part_{i + 1}.csv"
                chunk_path = os.path.join(save_folder_path, chunk_filename)

                df_chunk = pd.DataFrame(chunk_data, columns=cols)
                df_chunk.to_csv(chunk_path, index=False)
                saved_files_info.append(chunk_filename)
                print(f"Saved chunk {i + 1}/{num_chunks}: {chunk_path}")

            messagebox.showinfo("완료", f"{num_chunks}개의 파일이 다음 폴더에 저장되었습니다:\n{save_folder_path}")

        self._reset_conv_ui()

    def _reset_conv_ui(self):
        self.is_converting = False
        self.btn_start.config(state="normal")
        self.btn_pause.config(state="disabled")
        self.btn_stop.config(state="disabled")
        self.file_progress.set(0)
        self.total_progress.set(0)
        self.lbl_conv_status.config(text="대기 중")

    def start_training(self):
        csv_folder = self.csv_source_path.get()
        if not os.path.isdir(csv_folder):
            return messagebox.showwarning("경고", "학습할 CSV 파일이 담긴 폴더를 선택해주세요.")

        csv_paths = glob.glob(os.path.join(csv_folder, "*.csv"))
        if not csv_paths:
            return messagebox.showwarning("경고", "선택된 폴더에 CSV 파일이 없습니다.")

        try:
            epochs = int(self.ent_epochs.get())
            batch = int(self.ent_batch.get())
        except ValueError:
            return messagebox.showerror("오류", "Epochs와 Batch Size는 숫자만 입력하세요.")

        selected_engine = self.combo_trainer_engine.get()

        print(f"선택된 학습 엔진: {selected_engine}")
        if "LSTM" in selected_engine:
            self.current_trainer = ModelTrainer()
        elif "Transformer" in selected_engine:
            self.current_trainer = TransformerTrainer()
        else:
            return messagebox.showerror("오류", "알 수 없는 학습 엔진입니다.")

        self.is_training = True
        self.btn_train_start.config(state="disabled", text="🔥 학습 진행 중...", bg="#f1f3f4")

        # 단일 경로가 아닌, 파일 경로 리스트(list)를 전달
        threading.Thread(target=self._run_training, args=(csv_paths, epochs, batch), daemon=True).start()

    def _run_training(self, csv_path, epochs, batch):
        def progress_cb(epoch, logs):
            pct = (epoch / epochs) * 100
            acc, loss = logs.get('accuracy', 0), logs.get('loss', 0)
            msg = f"Epoch {epoch}/{epochs} | Acc: {acc:.4f} | Loss: {loss:.4f}"
            self.train_progress.set(pct)
            self.root.after(0, lambda: self.lbl_train_status.config(text=msg))
            print(f"[Train] {msg}")

        '''
        success, msg = self.trainer.train_model(csv_path, epochs, batch, progress_callback=progress_cb)
        self.root.after(0, lambda: self._finish_training(success, msg))
        '''
        if self.current_trainer:
            success, msg = self.current_trainer.train_model(
                csv_path, epochs, batch, progress_callback=progress_cb
            )
            self.root.after(0, lambda: self._finish_training(success, msg))
        else:
            self.root.after(0, lambda: self._finish_training(False, "Trainer가 초기화되지 않았습니다."))

    def _finish_training(self, success, msg):
        self.is_training = False
        self.btn_train_start.config(state="normal", text="🚀 학습 시작", bg="#ccffcc")
        self.tree_csv.config(selectmode="browse")
        self.train_progress.set(100)
        if success:
            self.lbl_train_status.config(text="학습 완료!", fg="green")
            messagebox.showinfo("성공", msg)
        else:
            self.lbl_train_status.config(text="오류 발생", fg="red")
            messagebox.showerror("실패", msg)

    def _browse_video(self):
        if self.is_testing: return
        filepath = filedialog.askopenfilename(title="테스트할 영상 선택", filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv"),
                                                                             ("All files", "*.*")))
        if filepath: self.video_path_test.set(filepath)

    def start_testing(self):
        yolo_model, lstm_model_name = self.combo_yolo_test.get(), self.combo_lstm_test.get()
        video_path = self.video_path_test.get()
        if "없음" in yolo_model or "없음" in lstm_model_name: return messagebox.showwarning("준비",
                                                                                        "YOLO와 LSTM 모델을 모두 선택해주세요.")
        if not os.path.exists(video_path): return messagebox.showwarning("준비", "테스트할 영상 파일을 먼저 불러와주세요.")
        lstm_model_path = os.path.join("trainer", "models", lstm_model_name)
        self.is_testing = True
        self.stop_test_event.clear()
        self.btn_test_start.config(state="disabled")
        self.btn_test_stop.config(state="normal")
        self.lbl_prediction.config(text="Initializing...", fg="blue")
        self.test_thread = threading.Thread(target=self._run_test, args=(video_path, yolo_model, lstm_model_path),
                                            daemon=True)
        self.test_thread.start()

    def stop_testing(self):
        if self.is_testing: self.stop_test_event.set()

    def _run_test(self, video_path, yolo_path, lstm_path):
        try:
            self.tester = VideoTester(yolo_path, lstm_path)
            self.root.after(0, self.lbl_prediction.config, {'text': "분석 시작...", 'fg': 'green'})

            cap = cv2.VideoCapture(video_path)
            while not self.stop_test_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                pred_label, display_frame = self.tester.process_frame(frame)

                h, w, _ = display_frame.shape
                max_h, max_w = self.lbl_test_preview.winfo_height(), self.lbl_test_preview.winfo_width()
                if max_w < 10 or max_h < 10: max_h, max_w = 600, 800
                scale = min(max_w / w, max_h / h)
                if scale < 1:
                    display_frame = cv2.resize(display_frame, (int(w * scale), int(h * scale)))

                rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))

                self.root.after(0, self._update_test_ui, pred_label, imgtk)
                time.sleep(0.01)

            cap.release()
        except Exception as e:
            error_details = traceback.format_exc()
            self.root.after(0, lambda msg=error_details: messagebox.showerror("테스트 오류", msg))
        finally:
            self.root.after(0, self._reset_test_ui)

    def _update_test_ui(self, label, imgtk):
        if not self.is_testing: return

        self.lbl_prediction.config(text=label)

        first_prediction = label.split('\n')[0]

        if any(keyword in first_prediction.lower() for keyword in ["punching", "theft", "pushing"]):
            self.lbl_prediction.config(fg="red")
        elif any(keyword in first_prediction.lower() for keyword in ["walking", "rotating"]):
            self.lbl_prediction.config(fg="orange")
        elif any(keyword in first_prediction.lower() for keyword in ["standing", "sitting"]):
            self.lbl_prediction.config(fg="green")
        else:
            self.lbl_prediction.config(fg="gray")

        self.lbl_test_preview.imgtk = imgtk
        self.lbl_test_preview.config(image=imgtk)

    def _reset_test_ui(self):
        self.is_testing = False
        self.btn_test_start.config(state="normal")
        self.btn_test_stop.config(state="disabled")
        self.lbl_prediction.config(text="대기 중", fg="gray")
        print("테스트가 중단되었습니다.")

    def on_close(self):
        if self.is_testing:
            self.stop_testing()
            if self.test_thread and self.test_thread.is_alive():
                self.test_thread.join(timeout=0.5)
        sys.stdout = self.original_stdout
        self.root.destroy()