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

    def train_model(self, csv_paths: list, epochs=50, batch_size=32, progress_callback=None):
        try:
            # 데이터 로딩 및 전처리
            df_list = [pd.read_csv(path) for path in csv_paths]
            df = pd.concat(df_list, ignore_index=True)

            feature_cols = [c for c in df.columns if c.startswith('v')]
            X = df[feature_cols].values
            y_integers, class_names = pd.factorize(df['label'])
            num_classes = len(class_names)
            y_cat = to_categorical(y_integers, num_classes=num_classes)

            num_samples, num_timesteps, num_features = X.shape[0], 30, 102
            X = X.reshape(num_samples, num_timesteps, num_features)

            # 데이터 증강 (단순 노이즈 증강만 사용)
            noise = np.random.normal(0, 0.01, X.shape)
            X_noisy = X + noise
            X_final = np.concatenate((X, X_noisy), axis=0)
            y_final = np.concatenate((y_cat, y_cat), axis=0)
            X_train, X_val, y_train, y_val = train_test_split(X_final, y_final, test_size=0.2, random_state=42)

            # 클래스 가중치 계산 (기존과 동일)
            y_train_integers = np.argmax(y_train, axis=1)
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers),
                                                              y=y_train_integers)
            class_weights_dict = dict(enumerate(class_weights))

            # 트랜스포머 모델 구성
            input_layer = Input(shape=(num_timesteps, num_features))

            # 포지셔널 인코딩 추가
            x = PositionalEncoding(position=num_timesteps, d_model=num_features)(input_layer)

            # 여러 개의 트랜스포머 블록을 쌓음
            for _ in range(2):  # 2-layer Transformer
                x = transformer_encoder_block(x, head_size=128, num_heads=4, ff_dim=128, dropout=0.1)

            # 최종 분류
            x = GlobalAveragePooling1D(data_format="channels_last")(x)
            x = Dropout(0.4)(x)
            x = Dense(64, activation="relu")(x)
            output_layer = Dense(num_classes, activation="softmax")(x)

            model = Model(inputs=input_layer, outputs=output_layer)

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.summary()

            # 콜백 및 학습 실행 (기존과 동일)
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True,
                                           verbose=1)  # patience 증가
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)  # patience 증가
            my_callbacks = [TrainingCallback(progress_callback), early_stopping, reduce_lr]

            history = model.fit(
                X_train, y_train, epochs=epochs, batch_size=batch_size,
                validation_data=(X_val, y_val), callbacks=my_callbacks,
                class_weight=class_weights_dict, verbose=0
            )

            # 결과 저장 및 반환
            architecture = "Transformer"
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_basename = f"{architecture}_model_{timestamp}"

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
            report_basename = model_basename
            report_txt_path = os.path.join(self.results_save_dir, f"{report_basename}.txt")
            with open(report_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
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

            # 시각화 자료 저장
            plot_save_path = os.path.join(self.results_save_dir, f"{report_basename}.png")
            self._plot_results(history, y_true, y_pred, class_names, plot_save_path)

            # 최종 메시지 반환
            stopped_epoch = early_stopping.stopped_epoch
            epoch_msg = f"(조기종료: {stopped_epoch + 1}/{epochs})" if stopped_epoch > 0 else f"({epochs}회 완료)"
            return True, f"학습 완료! ... (Transformer)\n저장됨: {model_save_path}"

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