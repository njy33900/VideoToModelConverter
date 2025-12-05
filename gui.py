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

# 모듈 임포트
from converter import VideoConverter
from trainer.train_logic import ModelTrainer


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

        self.root.title("AI Motion System (Converter & Trainer)")
        self.root.geometry("1100x800")

        # 인스턴스
        self.converter = None
        self.trainer = ModelTrainer()

        self.is_converting = False
        self.is_training = False

        self.preview_thread = None
        self.stop_preview_event = threading.Event()
        self.preview_lock = threading.Lock()

        self._init_layout()

        # 로그 리다이렉션
        self.redirector = TextRedirector(self.log_text)
        self.original_stdout = sys.stdout
        sys.stdout = self.redirector

        # 초기 로딩
        self._load_models_combo()
        self._load_video_tree()
        self._load_csv_list()

    def _init_layout(self):
        # 탭(Notebook) 구성
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # 탭 1: 데이터 변환
        self.tab_converter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_converter, text=" 데이터 변환 (Video -> CSV) ")
        self._init_converter_tab(self.tab_converter)

        # 탭 2: 모델 학습
        self.tab_trainer = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_trainer, text=" 모델 학습 (CSV -> AI Model) ")
        self._init_trainer_tab(self.tab_trainer)

        # 시스템 로그 창
        log_frame = ttk.LabelFrame(self.root, text="시스템 로그")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text = ScrolledText(log_frame, state='disabled', height=15, bg="black", fg="#00ff00",
                                     font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def on_tab_changed(self, event):
        self._stop_preview()

    # ========================================================
    # 탭 1: 데이터 변환기 UI
    # ========================================================
    def _init_converter_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 좌측: 파일 리스트
        left_frame = ttk.LabelFrame(paned, text="📂 영상 목록 (클릭하여 미리보기)")
        paned.add(left_frame, weight=1)

        scrollbar = ttk.Scrollbar(left_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_video = ttk.Treeview(left_frame, yscrollcommand=scrollbar.set)
        self.tree_video.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.tree_video.yview)

        self.tree_video.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.tree_video.bind("<Double-1>", self.on_tree_double_click)

        # 우측: 제어 패널
        right_frame = tk.Frame(paned)
        paned.add(right_frame, weight=3)

        # 모델 선택
        model_frame = ttk.LabelFrame(right_frame, text="YOLO 모델 선택")
        model_frame.pack(fill=tk.X, pady=5)
        self.combo_models = ttk.Combobox(model_frame, state="readonly")
        self.combo_models.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(model_frame, text="새로고침", command=self._load_models_combo).pack(side=tk.RIGHT, padx=5)

        # 미리보기
        preview_frame = ttk.LabelFrame(right_frame, text="미리보기")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.lbl_video = tk.Label(preview_frame, bg="black", text="영상 선택 시 재생", fg="white")
        self.lbl_video.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 제어 버튼
        ctrl_frame = ttk.LabelFrame(right_frame, text="변환 제어")
        ctrl_frame.pack(fill=tk.X, pady=5)

        self.conv_progress = tk.DoubleVar()
        ttk.Progressbar(ctrl_frame, variable=self.conv_progress, maximum=100).pack(fill=tk.X, padx=10, pady=5)
        self.lbl_conv_status = tk.Label(ctrl_frame, text="대기 중", fg="blue")
        self.lbl_conv_status.pack(pady=2)

        btn_box = tk.Frame(ctrl_frame)
        btn_box.pack(pady=5)
        self.btn_start = tk.Button(btn_box, text="▶ 전체 변환 시작", bg="#ccffcc", command=self.start_conversion)
        self.btn_start.pack(side=tk.LEFT, padx=5)
        self.btn_pause = tk.Button(btn_box, text="⏸ 일시정지", bg="#ffeb99", state="disabled", command=self.toggle_pause)
        self.btn_pause.pack(side=tk.LEFT, padx=5)
        self.btn_stop = tk.Button(btn_box, text="⏹ 중단", bg="#ffcccc", state="disabled", command=self.stop_conversion)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

    # ========================================================
    # 탭 2: 모델 학습기 UI
    # ========================================================
    def _init_trainer_tab(self, parent):
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 좌측: CSV 리스트
        left_frame = ttk.LabelFrame(paned, text="📊 학습 데이터셋 (CSV) 목록")
        paned.add(left_frame, weight=1)

        self.tree_csv = ttk.Treeview(left_frame, columns=("filename", "size", "date"), show="headings")
        self.tree_csv.heading("filename", text="파일명")
        self.tree_csv.heading("size", text="크기")
        self.tree_csv.heading("date", text="수정일")

        self.tree_csv.column("filename", width=250)
        self.tree_csv.column("size", width=80, anchor="center")
        self.tree_csv.column("date", width=120, anchor="center")

        self.tree_csv.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        btn_csv_refresh = ttk.Button(left_frame, text="목록 새로고침", command=self._load_csv_list)
        btn_csv_refresh.pack(fill=tk.X, padx=5, pady=5)

        # 우측: 학습 설정 및 제어
        right_frame = tk.Frame(paned)
        paned.add(right_frame, weight=1)

        # 설정
        setting_frame = ttk.LabelFrame(right_frame, text="학습 파라미터")
        setting_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(setting_frame, text="Epochs (반복 횟수):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ent_epochs = ttk.Entry(setting_frame)
        self.ent_epochs.insert(0, "50")
        self.ent_epochs.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(setting_frame, text="Batch Size (배치 크기):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ent_batch = ttk.Entry(setting_frame)
        self.ent_batch.insert(0, "32")
        self.ent_batch.grid(row=1, column=1, padx=5, pady=5)

        # 실행 버튼
        ctrl_frame = ttk.LabelFrame(right_frame, text="학습 제어")
        ctrl_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)

        self.train_progress = tk.DoubleVar()
        self.bar_train = ttk.Progressbar(ctrl_frame, variable=self.train_progress, maximum=100)
        self.bar_train.pack(fill=tk.X, padx=10, pady=20)

        self.lbl_train_status = tk.Label(ctrl_frame, text="CSV를 선택하고 학습을 시작하세요.", fg="gray", font=("Arial", 11))
        self.lbl_train_status.pack(pady=10)

        self.btn_train_start = tk.Button(ctrl_frame, text="🚀 학습 시작", bg="#ccffcc", font=("Arial", 14, "bold"), height=2,
                                         command=self.start_training)
        self.btn_train_start.pack(fill=tk.X, padx=20, pady=10)

    # --------------------------------------------------------
    # [공통] 로딩 함수들
    # --------------------------------------------------------
    def _load_models_combo(self):
        pt_files = glob.glob("*.pt")
        if not pt_files: pt_files = ["모델 없음"]
        self.combo_models['values'] = pt_files
        self.combo_models.current(0)

    def _load_video_tree(self):
        for item in self.tree_video.get_children(): self.tree_video.delete(item)
        if not os.path.exists(self.video_root): os.makedirs(self.video_root)

        for cls in self.class_map.keys():
            parent = self.tree_video.insert("", "end", text=f"📂 {cls}", open=True)
            path = os.path.join(self.video_root, cls)
            if os.path.exists(path):
                files = [f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                for f in files:
                    self.tree_video.insert(parent, "end", text=f, values=(os.path.join(path, f),))

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

    # --------------------------------------------------------
    # [탭 1] 기능 로직
    # --------------------------------------------------------
    def on_tree_select(self, event):
        if self.is_converting: return
        sel = self.tree_video.selection()
        if not sel: return
        item = self.tree_video.item(sel[0])
        if item['values']:
            self._start_panel_stream(item['values'][0])

    def on_tree_double_click(self, event):
        sel = self.tree_video.selection()
        if not sel: return
        item = self.tree_video.item(sel[0])
        if item['values']:
            path = item['values'][0]
            threading.Thread(target=self._popup_preview, args=(path,), daemon=True).start()

    def _stop_preview(self):
        self.stop_preview_event.set()
        if self.preview_thread and self.preview_thread.is_alive():
            self.preview_thread.join(timeout=0.2)
        self.stop_preview_event.clear()

    def _start_panel_stream(self, video_path):
        self._stop_preview()
        self.preview_thread = threading.Thread(target=self._stream_task, args=(video_path,), daemon=True)
        self.preview_thread.start()

    def _stream_task(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return

        MAX_W = 480
        MAX_H = 320

        while not self.stop_preview_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            h, w, _ = frame.shape

            scale = min(MAX_W / w, MAX_H / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            frame = cv2.resize(frame, (new_w, new_h))

            # BGR -> RGB 변환 및 Tkinter 이미지 생성
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))

            try:
                self.root.after(0, lambda i=imgtk: self._update_label(i))
            except:
                break
            time.sleep(0.033)
        cap.release()

    def _update_label(self, imgtk):
        self.lbl_video.imgtk = imgtk
        self.lbl_video.configure(image=imgtk, text="")

    def _popup_preview(self, path):
        cap = cv2.VideoCapture(path)
        win_name = f"Popup: {os.path.basename(path)}"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 640, 480)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow(win_name, frame)
            if cv2.waitKey(33) & 0xFF == ord('q'): break
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: break
        cap.release()
        cv2.destroyAllWindows()

    def start_conversion(self):
        model = self.combo_models.get()
        if "없음" in model: return messagebox.showerror("오류", "모델 없음")

        self.is_converting = True
        self._stop_preview()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.tree_video.config(selectmode="none")

        self.converter = VideoConverter(model_path=model, seq_length=30)
        threading.Thread(target=self._run_conversion, daemon=True).start()

    def stop_conversion(self):
        if self.converter and messagebox.askyesno("확인", "중단하시겠습니까?"):
            self.converter.stop_event.set()

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

    def _run_conversion(self):
        try:
            def cb(curr, total, name):
                pct = (curr / total) * 100
                self.conv_progress.set(pct)
                self.root.after(0, lambda: self.lbl_conv_status.config(text=f"처리 중: {name}"))

            dataset = self.converter.process_folder(self.video_root, self.class_map, progress_callback=cb)
            self.root.after(0, lambda: self._finish_conversion(dataset))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("에러", str(e)))
            self.root.after(0, self._reset_conv_ui)

    def _finish_conversion(self, dataset):
        if dataset:
            if not os.path.exists(self.save_root): os.makedirs(self.save_root)
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self.save_root, f"pose_data_{ts}.csv")

            # 메타데이터 컬럼 포함 (오류 수정됨)
            cols = [f'v{i}' for i in range(30 * 34)] + ['label', 'filename', 'frame', 'time']

            pd.DataFrame(dataset, columns=cols).to_csv(path, index=False)
            messagebox.showinfo("완료", f"저장됨:\n{path}")
            self._load_csv_list()
        self._reset_conv_ui()

    def _reset_conv_ui(self):
        self.is_converting = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.conv_progress.set(0)
        self.lbl_conv_status.config(text="대기 중")
        self.tree_video.config(selectmode="browse")

    # --------------------------------------------------------
    # [탭 2] 기능 로직
    # --------------------------------------------------------
    def start_training(self):
        sel = self.tree_csv.selection()
        if not sel:
            return messagebox.showwarning("경고", "CSV 선택 필요")

        filename = self.tree_csv.item(sel[0])['values'][0]
        csv_path = os.path.join(self.save_root, filename)

        try:
            epochs = int(self.ent_epochs.get())
            batch = int(self.ent_batch.get())
        except:
            return messagebox.showerror("오류", "숫자만 입력하세요")

        self.is_training = True
        self.btn_train_start.config(state="disabled", text="🔥 학습 진행 중...", bg="#f1f3f4")
        self.tree_csv.config(selectmode="none")
        self._stop_preview()

        threading.Thread(target=self._run_training, args=(csv_path, epochs, batch), daemon=True).start()

    def _run_training(self, csv_path, epochs, batch):
        def progress_cb(epoch, logs):
            pct = (epoch / epochs) * 100
            acc = logs.get('accuracy', 0)
            loss = logs.get('loss', 0)
            msg = f"Epoch {epoch}/{epochs} | Acc: {acc:.4f} | Loss: {loss:.4f}"
            self.train_progress.set(pct)
            self.root.after(0, lambda: self.lbl_train_status.config(text=msg))
            print(f"[Train] {msg}")

        success, msg = self.trainer.train_model(csv_path, epochs, batch, progress_callback=progress_cb)
        self.root.after(0, lambda: self._finish_training(success, msg))

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

    def on_close(self):
        self._stop_preview()
        sys.stdout = self.original_stdout
        self.root.destroy()