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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
import datetime
import json
import traceback

# 한글 폰트 설정 (Windows 기준)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 안정적인 데이터 증강 함수 (불안정한 Warp/Rotate 제외)
def augment_flip(sequence):
    """ 좌우 반전 증강 """
    seq_reshaped = sequence.copy().reshape(30, 17, 6)
    seq_reshaped[:, :, [0, 2, 4]] *= -1
    return seq_reshaped.reshape(30, 102)


def augment_scale(sequence):
    """ 스케일 조절 증강 """
    scale_factor = np.random.uniform(0.9, 1.1)
    return sequence * scale_factor

def augment_cutout(sequence):
    """ 컷아웃 증강 """
    seq_reshaped = sequence.copy().reshape(30, 17, 6)
    num_joints_to_cut = 2
    cutout_joints = np.random.choice(17, num_joints_to_cut, replace=False)
    seq_reshaped[:, cutout_joints, :] = 0
    return seq_reshaped.reshape(30, 102)

def augment_upper_body(sequence):
    """ 하체 가림(상반신만 보임) 증강 """
    seq_reshaped = sequence.copy().reshape(30, 17, 6)
    lower_body_indices = [11, 12, 13, 14, 15, 16]
    seq_reshaped[:, lower_body_indices, :] = 0
    return seq_reshaped.reshape(30, 102)

def augment_lower_body(sequence):
    """ 상반신 가림(하체만 보임) 증강 """
    seq_reshaped = sequence.copy().reshape(30, 17, 6)
    # 머리, 어깨, 팔꿈치, 손목 관절 인덱스 (0~10)
    upper_body_indices = list(range(11))
    seq_reshaped[:, upper_body_indices, :] = 0


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
        
    def train_model(self, csv_paths: list, epochs=50, batch_size=32, progress_callback=None):
        try:
            # 데이터 로딩 및 전처리
            print("--- [단계 1/5] 데이터 로딩 및 전처리 시작 ---")
            df_list = [pd.read_csv(p) for p in csv_paths]
            df = pd.concat(df_list, ignore_index=True)
            print(f"  - 총 {len(df)}개 데이터 로드 완료.")

            feature_cols = [c for c in df.columns if c.startswith('v')]
            X = df[feature_cols].values

            if np.isnan(X).any() or np.isinf(X).any():
                print("⚠️ 데이터에 NaN/Inf 발견. 0으로 치환합니다.")
                X = np.nan_to_num(X)

            y_integers, class_names = pd.factorize(df['label'])
            num_classes = len(class_names)
            y_cat = to_categorical(y_integers, num_classes=num_classes)
            print(f"  - {num_classes}개 클래스 감지 및 라벨 인코딩 완료.")

            num_samples, num_timesteps, num_features = X.shape[0], 30, 102
            X = X.reshape(num_samples, num_timesteps, num_features)
            print("--- [단계 1/5] 완료 ---\n")

            # 학습 / 검증 데이터 분리
            print("--- [단계 2/5] 학습/검증 데이터 분리 시작 ---")
            X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42)
            print(f"  - 원본 학습 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개.")
            print("--- [단계 2/5] 완료 ---\n")

            # 데이터 증강 파이프라인
            print("--- [단계 3/5] 데이터 증강 시작 ---")
            X_train_augmented_list = [X_train]
            y_train_augmented_list = [y_train]

            # 노이즈 증강
            print("  - 전체 데이터에 노이즈 증강 적용...")
            X_train_noisy = X_train + np.random.normal(0, 0.01, X_train.shape)
            X_train_augmented_list.append(X_train_noisy)
            y_train_augmented_list.append(y_train)

            # 하체 가림 증강
            print("  - 하체 가림(상반신) 데이터 증강 적용...")
            num_aug_upper = int(len(X_train) * 0.2)
            indices_upper = np.random.choice(len(X_train), num_aug_upper, replace=False)
            X_upper = np.array([augment_upper_body(X_train[idx]) for idx in indices_upper])
            y_upper = y_train[indices_upper]
            X_train_augmented_list.append(X_upper)
            y_train_augmented_list.append(y_upper)
            print(f"    - 하체 가림 데이터 {len(X_upper)}개 생성.")

            # 소수 클래스 집중 증강
            print("  - 소수 클래스('punching', 'pushing', 'reaching')에 추가 증강...")
            augmentation_targets = {"punching", "pushing", "reaching"}
            class_to_idx = {name: i for i, name in enumerate(class_names)}
            target_indices = {class_to_idx.get(name) for name in augmentation_targets if name in class_to_idx}

            y_train_indices = np.argmax(y_train, axis=1)
            minority_mask = np.isin(y_train_indices, list(target_indices))
            X_train_minority = X_train[minority_mask]
            y_train_minority = y_train[minority_mask]

            if len(X_train_minority) > 0:
                X_flipped = np.array([augment_flip(seq) for seq in X_train_minority])
                X_scaled = np.array([augment_scale(seq) for seq in X_train_minority])
                X_cutout = np.array([augment_cutout(seq) for seq in X_train_minority])

                X_train_augmented_list.extend([X_flipped, X_scaled, X_cutout])
                y_train_augmented_list.extend([y_train_minority, y_train_minority, y_train_minority])
                print(f"    - 소수 클래스 증강 데이터 {len(X_flipped) * 3}개 생성.")

            # 모든 데이터 병합
            X_train_final = np.concatenate(X_train_augmented_list)
            y_train_final = np.concatenate(y_train_augmented_list)
            print(f"✅ 증강 완료. 최종 학습 데이터 수: {len(X_train_final)}개")
            print("--- [단계 3/5] 완료 ---\n")

            # 이하 모든 코드의 들여쓰기를 바로잡았습니다

            # 클래스 가중치 계산
            print("--- [단계 4/5] 클래스 가중치 계산 시작 ---")
            y_train_integers_final = np.argmax(y_train_final, axis=1)
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers_final),
                                                              y=y_train_integers_final)
            class_weights_dict = dict(enumerate(class_weights))
            print("  - 클래스 불균형 조정을 위한 가중치 계산 완료.")
            print("--- [단계 4/5] 완료 ---\n")

            # 모델 구성
            print("--- [단계 5/5] LSTM 모델 구성 및 학습 시작 ---")
            input_layer = Input(shape=(num_timesteps, num_features))
            lstm_out = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(input_layer)
            lstm_out = Dropout(0.4)(lstm_out)
            attention_out = Attention(use_scale=True)([lstm_out, lstm_out])
            lstm_out_2 = Bidirectional(LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)))(attention_out)
            lstm_out_2 = Dropout(0.4)(lstm_out_2)
            dense_out = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(lstm_out_2)
            output_layer = Dense(num_classes, activation='softmax')(dense_out)
            model = Model(inputs=input_layer, outputs=output_layer)

            optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            history = model.fit(
                X_train_final, y_train_final,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=my_callbacks,
                class_weight=class_weights_dict,
                verbose=0
            )

            # 결과 저장 및 반환
            loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
            y_pred_probs = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val, axis=1)
            report = classification_report(y_true, y_pred, target_names=class_names)

            architecture = "LSTM"
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_basename = f"{architecture}_model_{timestamp}"

            model_save_path = os.path.join(self.model_save_dir, f"{model_basename}.h5")
            class_map_path = os.path.join(self.model_save_dir, f"{model_basename}_classes.json")
            model.save(model_save_path)
            with open(class_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_names.tolist(), f, ensure_ascii=False, indent=4)

            report_basename = model_basename
            report_txt_path = os.path.join(self.results_save_dir, f"{report_basename}.txt")
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Report - {datetime.datetime.now().strftime('%Y-%m-%d %H%M%S')}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Source CSVs ({len(csv_paths)} files):\n")
                for path in csv_paths:
                    f.write(f"  - {os.path.basename(path)}\n")
                f.write(f"\nSaved Model: {os.path.basename(model_save_path)}\n")
                f.write("\n--- Final Evaluation ---\n")
                f.write(f"Validation Loss: {loss:.4f}\n")
                f.write(f"Validation Accuracy: {final_acc:.4f}\n")
                f.write("\n--- Classification Report ---\n")
                f.write(report)
                f.write("\n--- Class Weights Used ---\n")
                f.write(json.dumps(class_weights_dict, indent=4))
            print(f"\n✅ 텍스트 리포트 저장됨: {report_txt_path}")

            plot_save_path = os.path.join(self.results_save_dir, f"{report_basename}.png")
            self._plot_results(history, y_true, y_pred, class_names, plot_save_path)

            stopped_epoch = early_stopping.stopped_epoch if hasattr(early_stopping,
                                                                    'stopped_epoch') and early_stopping.stopped_epoch > 0 else epochs
            epoch_msg = f"(조기종료: {stopped_epoch}/{epochs})" if hasattr(early_stopping,
                                                                       'stopped_epoch') and early_stopping.stopped_epoch > 0 else f"({epochs}회 완료)"
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