import numpy as np
import cv2
from typing import Optional


def fill_missing_keypoints(
        current_kp: np.ndarray,
        confidences: np.ndarray,
        last_valid_kp: np.ndarray
) -> np.ndarray:
    """
    낮은 신뢰도나 결측된 관절 좌표를 보정

    YOLO 추론 결과에서 신뢰도가 기준치(0.5) 미만이거나 좌표가 (0,0)인 경우,
    직전 프레임의 유효한 좌표값을 복사하여 데이터를 채움

    Args:
        current_kp (np.ndarray): 현재 프레임의 관절 좌표 배열 (Shape: 17x2)
        confidences (np.ndarray): 현재 프레임의 관절별 신뢰도 배열 (Shape: 17,)
        last_valid_kp (np.ndarray): 직전 프레임의 유효한 관절 좌표 배열 (Shape: 17x2)

    Returns:
        np.ndarray: 결측치가 이전 값으로 채워진 보정된 좌표 배열 (Shape: 17x2)
    """
    filled_kp = current_kp.copy()
    for i in range(17):
        # 신뢰도가 0.5 미만이거나 좌표가 (0,0)인 경우
        if confidences[i] < 0.5 or (filled_kp[i][0] == 0 and filled_kp[i][1] == 0):
            filled_kp[i] = last_valid_kp[i]
    return filled_kp


def get_stable_anchor(
        keypoints: np.ndarray,
        confidences: np.ndarray
) -> Optional[np.ndarray]:
    """
    CCTV 환경에 최적화된 안정적인 기준점(Anchor)을 계산

    하체 가림 현상이나 측면 자세에 대비하여 우선순위에 따라 기준점을 선정
    우선순위: 목(어깨 중점) > 골반(골반 중점) > 코(머리)

    Args:
        keypoints (np.ndarray): 보정된 관절 좌표 배열. (Shape: 17x2)
        confidences (np.ndarray): 관절 신뢰도 배열. (Shape: 17,)

    Returns:
        Optional[np.ndarray]: 기준점으로 사용할 [x, y] 좌표.
                              유효한 기준점을 찾지 못한 경우 None을 반환
    """
    # 1순위: 목 (양쪽 어깨의 중간) - 인덱스 5:왼어깨, 6:오른어깨
    if confidences[5] > 0.5 and confidences[6] > 0.5:
        return (keypoints[5] + keypoints[6]) / 2

    # 2순위: 골반 (양쪽 골반의 중간) - 인덱스 11:왼골반, 12:오른골반
    if confidences[11] > 0.5 and confidences[12] > 0.5:
        return (keypoints[11] + keypoints[12]) / 2

    # 3순위: 코 (머리) - 인덱스 0
    if confidences[0] > 0.5:
        return keypoints[0]

    return None


def normalize_to_relative(
        keypoints: np.ndarray,
        anchor: Optional[np.ndarray]
) -> np.ndarray:
    """
    절대 좌표를 기준점(Anchor) 중심의 상대 좌표로 변환 및 평탄화

    (현재 좌표 - 기준점 좌표) 연산을 수행하여, 카메라 내 피사체의 위치가 아닌
    '자세' 자체의 데이터만 추출

    Args:
        keypoints (np.ndarray): 보정된 관절 좌표 배열. (Shape: 17x2)
        anchor (Optional[np.ndarray]): 기준점(중심점) 좌표 [x, y]. None일 수 있음

    Returns:
        np.ndarray: 기준점을 뺀 후 1차원으로 평탄화(Flatten)된 좌표 배열. (Shape: 34,)
                    anchor가 None이면 원본 좌표를 평탄화하여 반환
    """
    if anchor is None:
        return keypoints.flatten()

    normalized = keypoints - anchor
    return normalized.flatten()

def letterbox_resize(image: np.ndarray, new_shape: tuple = (320, 240), color: tuple = (0, 0, 0)) -> np.ndarray:
    """
    원본 이미지의 가로세로 비율을 유지하면서 리사이즈하고, 남는 공간은 패딩으로 채움
    """
    shape = image.shape[:2]  # 현재 이미지 모양 [높이, 너비]
    new_w, new_h = new_shape

    # 스케일 비율 계산
    r = min(new_h / shape[0], new_w / shape[1])

    # 새로운 이미지 크기 계산
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_w - new_unpad[0]) / 2, (new_h - new_unpad[1]) / 2

    # 비율을 유지하며 리사이즈
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # 패딩 추가
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image