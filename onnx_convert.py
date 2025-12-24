import os
import h5py
import re

# Keras 3.0 간섭 차단
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
import tf_keras
import tf2onnx
import onnx
from onnxconverter_common import float16


def sanitize_keras3_h5(h5_path):
    """
    H5 파일 내부의 Keras 3 설정을 정제합니다.
    """
    print(f"🛠️  Keras 3 -> 2 호환성 패치 작업 시작: {h5_path}")
    try:
        with h5py.File(h5_path, 'r+') as f:
            if 'model_config' in f.attrs:
                config_raw = f.attrs['model_config']
                config = config_raw.decode('utf-8') if isinstance(config_raw, bytes) else config_raw

                # 1. batch_shape -> batch_input_shape
                config = config.replace('"batch_shape"', '"batch_input_shape"')

                # 2. DTypePolicy 객체 정제
                dtype_policy_pattern = r'\{"module":\s*"keras",\s*"class_name":\s*"DTypePolicy",\s*"config":\s*\{"name":\s*"float32"\},\s*"registered_name":\s*null\}'
                config = re.sub(dtype_policy_pattern, '"float32"', config)
                config = config.replace('"dtype_policy": "float32"', '"dtype": "float32"')

                # 3. registered_name 속성 제거
                config = config.replace(', "registered_name": null', '')

                f.attrs['model_config'] = config.encode('utf-8')
                print("✅ H5 패치 성공: 구조 정제 완료")
    except Exception as e:
        print(f"⚠️  패치 중 오류 발생(무시 가능): {e}")


def convert_h5_to_onnx(h5_path, output_onnx_path, use_fp16=True):
    if not os.path.exists(h5_path):
        print(f"❌ 파일을 찾을 수 없습니다: {h5_path}")
        return

    # 패치 실행
    sanitize_keras3_h5(h5_path)

    print(f"🔄 모델 로드 중 (Inference Only): {h5_path}")
    try:
        # [핵심 변경] compile=False 를 사용하여 옵티마이저 로딩 에러 방지
        model = tf_keras.models.load_model(h5_path, compile=False)
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 로드 최종 실패: {e}")
        return

    # 모델의 입력 형태 정의 (사용자 모델 스펙)
    spec = (tf.TensorSpec((None, 30, 102), tf.float32, name="input"),)

    print("🚀 ONNX 변환 시작 (FP32)...")
    # tf2onnx 변환 실행
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

    onnx.save(model_proto, output_onnx_path)
    print(f"✅ 기본 ONNX 저장 완료: {output_onnx_path}")

    if use_fp16:
        print("⚡ FP16 양자화 적용 중...")
        fp16_path = output_onnx_path.replace(".onnx", "_fp16.onnx")
        onnx_model = onnx.load(output_onnx_path)
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, fp16_path)
        print(f"✅ FP16 모델 저장 완료: {fp16_path}")
        return fp16_path

    return output_onnx_path


if __name__ == "__main__":
    # 경로를 실제 환경에 맞게 수정하세요
    input_h5 = "trainer/models/LSTM_model_20251219_175600.h5"
    output_onnx = "action_model_v0.95.onnx"
    convert_h5_to_onnx(input_h5, output_onnx)