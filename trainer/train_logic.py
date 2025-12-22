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

# 한글 폰트 설정
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

    # ----------------------------------------------------------------
    # 데이터 증강(Augmentation) 함수들
    # ----------------------------------------------------------------
    def _augment_time_warp(self, X, sigma=0.2, knots=4):
        """Time Warping: 동작 속도 변화"""
        time_steps = X.shape[0]
        feature_dim = X.shape[1]

        warp_steps = np.arange(time_steps)
        ctrl_pts = np.linspace(0, time_steps - 1, knots)
        time_ctrl = ctrl_pts + np.random.normal(0, sigma, size=knots) * (time_steps / knots)
        time_ctrl = np.clip(time_ctrl, 0, time_steps - 1)
        time_ctrl[0] = 0
        time_ctrl[-1] = time_steps - 1

        new_time_indices = np.interp(warp_steps, ctrl_pts, time_ctrl)

        X_warped = np.zeros_like(X)
        for i in range(feature_dim):
            X_warped[:, i] = np.interp(new_time_indices, warp_steps, X[:, i])
        return X_warped

    def _augment_scaling(self, X, sigma=0.1):
        """Scaling: 크기 변화"""
        factor = np.random.normal(1.0, sigma)
        return X * factor

    def _augment_time_shift(self, X, shift_range=5):
        """Time Shift: 시점 변화"""
        shift = np.random.randint(-shift_range, shift_range)
        if shift == 0: return X

        X_shifted = np.zeros_like(X)
        if shift > 0:
            X_shifted[shift:] = X[:-shift]
            X_shifted[:shift] = X[0]
        else:
            shift = abs(shift)
            X_shifted[:-shift] = X[shift:]
            X_shifted[-shift:] = X[-1]
        return X_shifted

    def _augment_joint_dropout(self, X, drop_prob=0.05):
        """Joint Dropout: 센서 튐/가림 현상 모사"""
        mask = np.random.choice([0, 1], size=X.shape, p=[drop_prob, 1 - drop_prob])
        return X * mask

    def _augment_noise(self, X, sigma=0.01):
        """Noise: 기본 잡음 추가"""
        noise = np.random.normal(0, sigma, X.shape)
        return X + noise

    # ----------------------------------------------------------------
    # 메인 학습 로직
    # ----------------------------------------------------------------
    def train_model(self, csv_paths: list, epochs=50, batch_size=32, progress_callback=None):
        try:
            # 1. 데이터 로딩
            print(f"DEBUG: {len(csv_paths)}개의 CSV 파일 로딩 중...")
            df_list = [pd.read_csv(path) for path in csv_paths]
            if not df_list: return False, "CSV 파일이 없습니다."

            df = pd.concat(df_list, ignore_index=True)
            print(f"✅ 총 {len(df)}개의 원본 데이터 로드 완료.")

            feature_cols = [c for c in df.columns if c.startswith('v')]
            X_raw = df[feature_cols].values
            y_integers_raw, class_names = pd.factorize(df['label'])

            num_samples = X_raw.shape[0]
            num_timesteps = 30
            num_features = 102  # (Pos 34 + Vel 34 + Acc 34)

            if X_raw.shape[1] != num_timesteps * num_features:
                raise ValueError("CSV 특징 수 오류")

            X_raw = X_raw.reshape(num_samples, num_timesteps, num_features)

            # One-hot Encoding
            num_classes = len(class_names)
            y_cat_raw = to_categorical(y_integers_raw, num_classes=num_classes)

            # ------------------------------------------------------------
            # 2. 먼저 분할 (Split First) -> Data Leakage 방지
            # ------------------------------------------------------------
            print("\n✂️ 데이터 분할 중 (검증 데이터 오염 방지)...")
            # Stratify로 클래스 비율 유지하며 분할
            X_train_raw, X_val, y_train_raw, y_val = train_test_split(
                X_raw, y_cat_raw, test_size=0.2, random_state=42, stratify=y_cat_raw
            )

            # ------------------------------------------------------------
            # 3. 학습 데이터만 증강 (Augment Train Only)
            # ------------------------------------------------------------
            print(f"✨ Train 데이터 증강 시작 (원본 학습 데이터: {len(X_train_raw)}개)")

            # 증강 로직을 위해 one-hot을 다시 인덱스로 변환 (임시)
            y_train_indices = np.argmax(y_train_raw, axis=1)

            # 클래스별 증강 전략 설정
            rare_classes = ['punching', 'pushing', 'reaching']  # 부족한 클래스
            abundant_classes = ['standing', 'etc', 'sitting']  # 충분한 클래스 (증강 X)

            X_train_final_list = [X_train_raw]  # 원본 Train 데이터 포함
            y_train_final_list = [y_train_indices]

            class_map = {name: i for i, name in enumerate(class_names)}

            for cls_name in class_names:
                cls_idx = class_map[cls_name]
                indices = np.where(y_train_indices == cls_idx)[0]
                X_subset = X_train_raw[indices]

                # 증강 배수 결정
                if cls_name in rare_classes:
                    multiplier = 20  # 부족한 데이터: 5배 증강
                    status = "부족/강력증강"
                elif cls_name in abundant_classes:
                    multiplier = 1  # 충분한 데이터: 증강 안 함
                    status = "충분/증강제외"
                else:
                    multiplier = 5  # 일반 데이터: 1배 증강
                    status = "일반/1배증강"

                print(f"  -> 클래스 '{cls_name}': {len(X_subset)}개 ({status})")

                if len(X_subset) == 0 or multiplier == 0:
                    continue

                # 증강 루프
                for _ in range(multiplier):
                    X_aug_batch = []
                    for sample in X_subset:
                        aug_sample = sample.copy()

                        # 확률적 증강 적용
                        if np.random.rand() > 0.5: aug_sample = self._augment_scaling(aug_sample, sigma=0.05)
                        if np.random.rand() > 0.5: aug_sample = self._augment_time_warp(aug_sample, sigma=0.2)
                        if np.random.rand() > 0.7: aug_sample = self._augment_time_shift(aug_sample, shift_range=5)
                        if np.random.rand() > 0.5: aug_sample = self._augment_joint_dropout(aug_sample, drop_prob=0.05)

                        # 노이즈는 항상 적용
                        aug_sample = self._augment_noise(aug_sample, sigma=0.01)
                        X_aug_batch.append(aug_sample)

                    X_train_final_list.append(np.array(X_aug_batch))
                    y_train_final_list.append(np.full(len(X_subset), cls_idx))

            # Train 데이터 병합
            X_train = np.concatenate(X_train_final_list, axis=0)
            y_train_int = np.concatenate(y_train_final_list, axis=0)
            y_train = to_categorical(y_train_int, num_classes=num_classes)

            print(f"✅ 증강 완료: Train {len(X_train_raw)} -> {len(X_train)} 샘플")
            print(f"✅ Validation 데이터: {len(X_val)} 샘플 (증강 X, 원본 유지)")

            # 분포 확인
            unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
            dist_dict = dict(zip([class_names[i] for i in unique], counts))
            print(f"📊 최종 Train 데이터 분포: {dist_dict}")

            # 클래스 가중치 계산 (증강 후 데이터 기준)
            y_train_integers = np.argmax(y_train, axis=1)
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced',
                classes=np.unique(y_train_integers),
                y=y_train_integers
            )
            class_weights_dict = dict(enumerate(class_weights))
            print(f"⚖️ 클래스 가중치 적용: {class_weights_dict}")

            # ------------------------------------------------------------
            # 모델 구성 (LSTM)
            # ------------------------------------------------------------
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)),
                              input_shape=(num_timesteps, num_features)),
                Dropout(0.4),
                Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(0.001))),
                Dropout(0.4),
                Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # ------------------------------------------------------------
            # 학습 실행
            # ------------------------------------------------------------
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            # Validation data는 순수한 X_val, y_val 사용
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=my_callbacks,
                class_weight=class_weights_dict,
                verbose=0
            )

            # ------------------------------------------------------------
            # 결과 저장
            # ------------------------------------------------------------
            architecture = "LSTM"
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_basename = f"{architecture}_model_{timestamp}"

            model_save_path = os.path.join(self.model_save_dir, f"{model_basename}.h5")
            class_map_path = os.path.join(self.model_save_dir, f"{model_basename}_classes.json")

            model.save(model_save_path)
            with open(class_map_path, 'w', encoding='utf-8') as f:
                json.dump(class_names.tolist(), f, ensure_ascii=False, indent=4)

            # 최종 평가 (순수 Validation 셋 사용)
            loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
            y_pred_probs = model.predict(X_val, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true = np.argmax(y_val, axis=1)
            report = classification_report(y_true, y_pred, target_names=class_names)

            # 텍스트 리포트 저장
            report_basename = model_basename
            report_txt_path = os.path.join(self.results_save_dir, f"{report_basename}.txt")
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n")
                f.write(f"Source CSVs ({len(csv_paths)} files):\n")
                for path in csv_paths:
                    f.write(f"  - {os.path.basename(path)}\n")
                f.write(f"\nSaved Model: {os.path.basename(model_save_path)}\n")
                f.write(f"Augmentation Strategy: Split First -> Augment Train Only\n")
                f.write("\n--- Final Evaluation (Clean Validation Set) ---\n")
                f.write(f"Validation Loss: {loss:.4f}\n")
                f.write(f"Validation Accuracy: {final_acc:.4f}\n")
                f.write("\n--- Classification Report ---\n")
                f.write(report)
                f.write("\n--- Class Weights Used ---\n")
                f.write(json.dumps(class_weights_dict, indent=4))
                f.write("\n--- Train Data Distribution After Augmentation ---\n")
                f.write(json.dumps({str(k): int(v) for k, v in dist_dict.items()}, indent=4, ensure_ascii=False))

            print(f"\n✅ 텍스트 리포트 저장됨: {report_txt_path}")

            # 시각화 자료 저장
            plot_save_path = os.path.join(self.results_save_dir, f"{report_basename}.png")
            self._plot_results(history, y_true, y_pred, class_names, plot_save_path)

            stopped_epoch = early_stopping.stopped_epoch
            epoch_msg = f"(조기종료: {stopped_epoch + 1}/{epochs})" if stopped_epoch > 0 else f"({epochs}회 완료)"
            return True, f"학습 완료! {epoch_msg} (LSTM - Clean Val)\n검증 정확도: {final_acc:.4f}\n저장됨: {model_save_path}"

        except Exception as e:
            traceback.print_exc()
            return False, f"오류 발생: {str(e)}"

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
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('오답 분석 (Confusion Matrix)')
        plt.ylabel('실제값')
        plt.xlabel('예측값')
        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path)
            except:
                pass
        plt.show()