import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import Callback
import datetime


class TrainingCallback(Callback):
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback:
            # logs: {'loss': ..., 'accuracy': ...}
            self.progress_callback(epoch + 1, logs)


class ModelTrainer:
    def __init__(self):
        self.model_save_dir = os.path.join("trainer", "models")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def train_model(self, csv_path, epochs=50, batch_size=32, progress_callback=None):
        """
        CSV 파일을 읽어 LSTM 모델 학습
        """
        try:
            # 데이터 로드
            print(f"DEBUG: 데이터 로딩 중... {csv_path}")
            df = pd.read_csv(csv_path)

            # 메타데이터 컬럼 제거 (label, filename, frame, time 등 제외하고 v0~v1019만 사용)
            # v로 시작하는 컬럼만 선택
            feature_cols = [c for c in df.columns if c.startswith('v')]

            X = df[feature_cols].values
            y = df['label'].values

            # 데이터 전처리
            # (Samples, 30 frames, 34 features)
            # 30 * 34 = 1020 features
            num_samples = X.shape[0]
            X = X.reshape(num_samples, 30, 34)

            # One-hot Encoding
            num_classes = len(np.unique(y))
            y = to_categorical(y, num_classes=3)

            # 학습/검증 분리
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # 모델 정의
            model = Sequential([
                LSTM(64, return_sequences=False, input_shape=(30, 34)),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # [Neutral, Movement, Threat]
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # 학습 실행
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[TrainingCallback(progress_callback)],
                verbose=0  # 콘솔 출력 끄기 (GUI로 대체)
            )

            # 모델 저장
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"lstm_model_{timestamp}.h5"  # .keras 권장되나 호환성 위해 .h5
            save_path = os.path.join(self.model_save_dir, model_filename)
            model.save(save_path)

            final_acc = history.history['accuracy'][-1]
            return True, f"학습 완료!\n정확도: {final_acc:.4f}\n저장됨: {save_path}"

        except Exception as e:
            return False, str(e)