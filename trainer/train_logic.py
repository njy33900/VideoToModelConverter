import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import datetime
import json
import traceback

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
        self.results_save_dir = os.path.join("trainer", "results")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

    def train_model(self, csv_path, epochs=50, batch_size=32, progress_callback=None):
        try:
            # 데이터 로딩 및 전처리
            print(f"DEBUG: 데이터 로딩 중... {csv_path}")
            df = pd.read_csv(csv_path)
            feature_cols = [c for c in df.columns if c.startswith('v')]
            X = df[feature_cols].values
            y_integers, class_names = pd.factorize(df['label'])
            num_classes = len(class_names)
            y_cat = to_categorical(y_integers, num_classes=num_classes)
            X = X.reshape(X.shape[0], 30, 34)

            # 데이터 증강
            print("✨ 데이터 증강 적용 중 (Noise Injection)...")
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise
            X_final = np.concatenate((X, X_noisy), axis=0)
            y_final = np.concatenate((y_cat, y_cat), axis=0)
            X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
            print(f"✅ 증강 완료: {len(X)} -> {len(X_final)} 샘플")

            # 클래스 가중치 계산
            y_train_integers = np.argmax(y_train, axis=1)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_integers),
                y=y_train_integers
            )
            class_weights_dict = dict(enumerate(class_weights))
            print(f"⚖️ 클래스 가중치 적용: {class_weights_dict}")

            # 모델 구성
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)), input_shape=(30, 34)),
                Dropout(0.4),
                Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001))),
                Dropout(0.4),
                Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # 콜백 및 학습 실행
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=my_callbacks,
                class_weight=class_weights_dict,
                verbose=0
            )

            # 결과 저장 로직
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_basename = f"lstm_model_{timestamp}"

            model_save_path = os.path.join(self.model_save_dir, f"{model_basename}.h5")
            class_map_path = os.path.join(self.model_save_dir, f"{model_basename}_classes.json")
            model.save(model_save_path)
            with open(class_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_names.tolist(), f, ensure_ascii=False, indent=4)

            # 최종 평가
            loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
            y_pred_probs = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val, axis=1)
            report = classification_report(y_true, y_pred, target_names=class_names)

            # 텍스트 리포트 저장
            report_basename = f"training_report_{timestamp}"
            report_txt_path = os.path.join(self.results_save_dir, f"{report_basename}.txt")
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Source CSV: {os.path.basename(csv_path)}\n")
                f.write(f"Saved Model: {os.path.basename(model_save_path)}\n")
                f.write("\n--- Final Evaluation ---\n")
                f.write(f"Validation Loss: {loss:.4f}\n")
                f.write(f"Validation Accuracy: {final_acc:.4f}\n")
                f.write("\n--- Classification Report ---\n")
                f.write(report)
                f.write("\n--- Class Weights Used ---\n")
                f.write(json.dumps(class_weights_dict, indent=4))
            print(f"\n✅ 텍스트 리포트 저장됨: {report_txt_path}")

            # 시각화 자료 저장
            plot_save_path = os.path.join(self.results_save_dir, f"{report_basename}.png")
            self._plot_results(history, y_true, y_pred, class_names, plot_save_path)

            stopped_epoch = early_stopping.stopped_epoch
            epoch_msg = f"(조기종료: {stopped_epoch + 1}/{epochs})" if stopped_epoch > 0 else f"({epochs}회 완료)"
            return True, f"학습 완료! {epoch_msg}\n검증 정확도: {final_acc:.4f}\n저장됨: {model_save_path}"

        except Exception as e:
            traceback.print_exc()
            return False, str(e)

    def _plot_results(self, history, y_true, y_pred, class_names, save_path=None):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.plot(history.history['loss'], label='Train Loss', linestyle='--')
        plt.plot(history.history['val_loss'], label='Val Loss', linestyle='--')
        plt.title('학습 진행 상황')
        plt.xlabel('Epochs')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.title('오답 분석 (Confusion Matrix)')
        plt.ylabel('실제값')
        plt.xlabel('예측값')

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
                print(f"✅ 시각화 자료 저장됨: {save_path}")
            except Exception as e:
                print(f"오류: 시각화 자료 저장 실패 - {e}")

        plt.show()