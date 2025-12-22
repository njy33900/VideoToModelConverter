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
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
                                     GlobalAveragePooling1D, Add)
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical, register_keras_serializable

import datetime
import json
import traceback

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False


@register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    """
    입력 시퀀스에 위치 정보를 추가하는 레이어.
    """

    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
        })
        return config


def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Multi-Head Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = Add()([x, inputs])

    # Feed Forward Network
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return Add()([x, res])


class TrainingCallback(Callback):
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        if self.progress_callback: self.progress_callback(epoch + 1, logs)


# TransformerTrainer 클래스
class TransformerTrainer:
    def __init__(self):
        self.model_save_dir = os.path.join("trainer", "models")
        self.results_save_dir = os.path.join("trainer", "results")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.results_save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 데이터 증강(Augmentation) 함수들 (LSTM 버전과 동일하게 이식)
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
            df = pd.concat(df_list, ignore_index=True)
            print(f"✅ 총 {len(df)}개의 원본 데이터 로드 완료.")

            feature_cols = [c for c in df.columns if c.startswith('v')]
            X_raw = df[feature_cols].values
            y_integers_raw, class_names = pd.factorize(df['label'])

            num_samples = X_raw.shape[0]
            num_timesteps = 30
            num_features = 102

            if X_raw.shape[1] != num_timesteps * num_features:
                raise ValueError("CSV 특징 수 오류")

            X_raw = X_raw.reshape(num_samples, num_timesteps, num_features)

            # One-hot Encoding
            num_classes = len(class_names)
            y_cat_raw = to_categorical(y_integers_raw, num_classes=num_classes)

            # ------------------------------------------------------------
            # 먼저 분할 (Split First)
            # ------------------------------------------------------------
            print("\n✂️ 데이터 분할 중 (먼저 나누고 Train만 증강합니다)...")
            # stratify를 사용하여 클래스 비율을 유지하며 나눔
            X_train_raw, X_val, y_train_raw, y_val = train_test_split(
                X_raw, y_cat_raw, test_size=0.2, random_state=42, stratify=y_cat_raw
            )

            # ------------------------------------------------------------
            # Train 데이터만 증강 (Augment Train Only)
            # ------------------------------------------------------------
            print(f"✨ Train 데이터 증강 시작 (원본 학습 데이터: {len(X_train_raw)}개)")

            # y_train_raw는 one-hot 상태이므로 다시 정수형 인덱스로 변환 (증강 로직 위해)
            y_train_indices = np.argmax(y_train_raw, axis=1)

            rare_classes = ['punching', 'pushing', 'reaching']
            abundant_classes = ['standing', 'etc', 'sitting']

            X_train_final_list = [X_train_raw]  # 원본 Train 포함
            y_train_final_list = [y_train_indices]

            class_map = {name: i for i, name in enumerate(class_names)}

            for cls_name in class_names:
                cls_idx = class_map[cls_name]
                # Train 데이터 중에서 해당 클래스만 찾음
                indices = np.where(y_train_indices == cls_idx)[0]
                X_subset = X_train_raw[indices]

                # 증강 배수 결정
                if cls_name in rare_classes:
                    multiplier = 5  # 부족: 5배
                elif cls_name in abundant_classes:
                    multiplier = 0  # 충분: 증강 안 함
                else:
                    multiplier = 1  # 일반: 1배

                if len(X_subset) == 0 or multiplier == 0:
                    continue

                for _ in range(multiplier):
                    X_aug_batch = []
                    for sample in X_subset:
                        aug_sample = sample.copy()

                        # 랜덤 증강 적용
                        if np.random.rand() > 0.5: aug_sample = self._augment_scaling(aug_sample, sigma=0.05)
                        if np.random.rand() > 0.5: aug_sample = self._augment_time_warp(aug_sample, sigma=0.2)
                        if np.random.rand() > 0.7: aug_sample = self._augment_time_shift(aug_sample, shift_range=5)
                        if np.random.rand() > 0.5: aug_sample = self._augment_joint_dropout(aug_sample, drop_prob=0.05)
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

            # 클래스 가중치 계산
            y_train_integers = np.argmax(y_train, axis=1)
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers),
                                                              y=y_train_integers)
            class_weights_dict = dict(enumerate(class_weights))

            # ------------------------------------------------------------
            # 모델 구성 (Transformer) - 기존과 동일
            # ------------------------------------------------------------
            input_layer = Input(shape=(num_timesteps, num_features))
            x = PositionalEncoding(position=num_timesteps, d_model=num_features)(input_layer)
            for _ in range(2):
                x = transformer_encoder_block(x, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)
            x = GlobalAveragePooling1D(data_format="channels_last")(x)
            x = Dropout(0.4)(x)
            x = Dense(64, activation="relu")(x)
            output_layer = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # 학습 실행
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            history = model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, y_val),  # 검증은 원본 데이터로!
                callbacks=my_callbacks,
                class_weight=class_weights_dict, verbose=0
            )

            # (이하 결과 저장 코드는 기존과 동일)
            # ... [기존 코드의 결과 저장 부분 복사] ...
            # 여기서는 편의상 생략합니다. 기존 코드의 뒷부분을 그대로 쓰시면 됩니다.

            # --- 임시 리턴 (기존 코드에 붙일 때 삭제하세요) ---
            loss, final_acc = model.evaluate(X_val, y_val, verbose=0)
            return True, f"학습 완료! (Split First 적용됨)\n검증 정확도: {final_acc:.4f}"

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