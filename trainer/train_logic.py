import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# [추가됨] 클래스 불균형 해소용 라이브러리
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# [추가됨] 과적합 방지용 규제 라이브러리
from tensorflow.keras.regularizers import l2
import datetime

# 한글 폰트 설정 (Windows 기준)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


class TrainingCallback(Callback):
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback:
            self.progress_callback(epoch + 1, logs)


class ModelTrainer:
    def __init__(self):
        self.model_save_dir = os.path.join("trainer", "models")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

    def train_model(self, csv_path, epochs=50, batch_size=32, progress_callback=None):
        try:
            print(f"DEBUG: 데이터 로딩 중... {csv_path}")
            df = pd.read_csv(csv_path)

            feature_cols = [c for c in df.columns if c.startswith('v')]
            X = df[feature_cols].values
            y = df['label'].values

            num_samples = X.shape[0]
            X = X.reshape(num_samples, 30, 34)

            num_classes = len(np.unique(y))
            y_cat = to_categorical(y, num_classes=num_classes)

            # ------------------------------------------------------------
            # 1. 데이터 증강 (Data Augmentation) - 기존 유지
            # ------------------------------------------------------------
            print("✨ 데이터 증강 적용 중 (Noise Injection)...")
            X_aug = X.copy()
            y_aug = y_cat.copy()

            # 노이즈 추가 (표준편차 0.01)
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise

            # 데이터 합치기 (2배)
            X_final = np.concatenate((X, X_noisy), axis=0)
            y_final = np.concatenate((y_cat, y_cat), axis=0)
            print(f"✅ 증강 완료: {len(X)} -> {len(X_final)} 샘플")

            # 학습/검증 분리
            X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

            # ------------------------------------------------------------
            # [추가됨] 클래스 가중치 계산 (Class Weight)
            # 데이터가 적은 클래스(Movement 등)를 틀리면 더 큰 페널티를 부여
            # ------------------------------------------------------------
            y_integers = np.argmax(y_train, axis=1)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_integers),
                y=y_integers
            )
            class_weights_dict = dict(enumerate(class_weights))
            print(f"⚖️ 클래스 가중치 적용: {class_weights_dict}")

            # ============================================================
            # [수정됨] 모델 구조 강화: L2 규제 추가 + 드롭아웃 상향
            # ============================================================
            model = Sequential([
                # 1. 양방향 LSTM + L2 규제 (가중치가 너무 커지는 것을 방지)
                Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=(30, 34)),
                Dropout(0.4),  # 과적합 방지를 위해 0.3 -> 0.4 상향

                # 2. Stacked LSTM + L2 규제
                Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
                Dropout(0.4),  # 드롭아웃 상향

                # 3. 분류기
                Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # 콜백 설정
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            # 학습 실행 (class_weight 추가)
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=my_callbacks,
                class_weight=class_weights_dict,  # [핵심] 가중치 적용
                verbose=0
            )

            # 저장
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"lstm_model_{timestamp}.h5"
            save_path = os.path.join(self.model_save_dir, model_filename)
            model.save(save_path)

            # ============================================================
            # 검증 및 시각화 (기존 유지)
            # ============================================================
            loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
            print(f"\n📊 [최종 평가] Loss: {loss:.4f} | Accuracy: {final_acc:.4f}")

            y_pred_probs = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val, axis=1)

            report = classification_report(y_true, y_pred, target_names=["Neutral", "Movement", "Threat"])
            print("\n📋 [상세 분류 리포트]")
            print(report)

            self._plot_results(history, y_true, y_pred)

            stopped_epoch = early_stopping.stopped_epoch
            epoch_msg = f"(조기종료: {stopped_epoch + 1}/{epochs})" if stopped_epoch > 0 else f"({epochs}회 완료)"

            return True, f"학습 완료! {epoch_msg}\n검증 정확도: {final_acc:.4f}\n저장됨: {save_path}"

        except Exception as e:
            return False, str(e)

    def _plot_results(self, history, y_true, y_pred):
        plt.figure(figsize=(12, 5))

        # 1. 학습 곡선
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
        plt.title('학습 진행 상황 (규제 적용됨)')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        # 2. 혼동 행렬
        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Neutral", "Movement", "Threat"],
                    yticklabels=["Neutral", "Movement", "Threat"])
        plt.title('오답 분석 (Confusion Matrix)')
        plt.ylabel('실제값')
        plt.xlabel('예측값')

        plt.tight_layout()
        plt.show()