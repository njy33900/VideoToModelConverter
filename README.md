## 개요

YOLO Pose Estimation을 이용해 영상에서 사람의 관절 데이터를 추출하고,  
LSTM(Long Short-Term Memory) 신경망을 통해 행동(정지, 이동, 위협)을 분류/학습하는 올인원 시스템입니다.

## 📂 프로젝트 구조
```txt
VideoToCSV/
├── .venv/                 # 가상환경
├── analyzed/              # 변환 완료된 CSV 데이터 저장소
├── logs/                  # 시스템 로그
├── raw_videos/            # 학습용 원본 영상 폴더
│   ├── movement/          # [Class 1] 이동 행동 영상
│   ├── neutral/           # [Class 0] 정지/평상시 영상
│   └── threat/            # [Class 2] 위협/이상 행동 영상
├── trainer/               # 학습 관련 모듈
│   ├── models/            # 학습 완료된 모델(.h5) 저장소
│   └── train_logic.py     # LSTM 모델 정의 및 학습 로직
├── converter.py           # 영상 -> CSV 변환 로직 (YOLO 추론)
├── gui.py                 # Tkinter 기반 GUI 구성
├── main.py                # 프로그램 진입점 (Entry Point)
├── pose_utils.py          # 관절 좌표 보정 및 정규화 알고리즘
└── .gitignore
```

## 설치 및 환경 설정
### 요구 사항 (Prerequisites)
Python 3.8+  
NVIDIA GPU (권장, CUDA 설정 필요)

### 설치
저장소를 클론합니다.
```
git clone https://github.com/njy33900/VideoToModelConverter.git
```
필수 라이브러리를 설치합니다.
```
pip install opencv-python ultralytics pandas numpy tensorflow scikit-learn pillow
```

## 🚀 사용 방법 (Usage)
### 1. 데이터 준비
raw_videos 폴더 내에 클래스별로 영상을 넣습니다.
- neutral: 가만히 서 있거나, 핸드폰 보기, 뒷짐 지기 등. ※ 발이 떨어지지 않는 자세들
- movement: 걷기, 뛰기, 물건 들고 이동하기 등. ※ 발이 움직이는 자세들
- threat: 손 뻗기, 주먹질, 쓰러짐 등 위협적인 행동.
### 2. 프로그램 실행
run.bat을 실행합니다.

## 📜 License
This project is licensed under the MIT License.
