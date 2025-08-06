# dlproject
# 😷 마스크 착용 감지기 (ResNet18 기반)

얼굴 이미지를 분석하여 마스크 착용 여부를 자동으로 판별하는 딥러닝 프로젝트입니다.

## 📋 프로젝트 개요

COVID-19 시대에 필수가 된 마스크 착용 여부를 AI가 자동으로 감지합니다.
사용자가 이미지를 업로드하면 **마스크 착용** 또는 **마스크 미착용**으로 분류해줍니다.

### 🎯 주요 기능
- 😷 마스크 착용/미착용 자동 분류
- 📊 실시간 예측 결과 및 신뢰도 표시
- 🌐 웹 기반 사용자 인터페이스
- 🚀 빠른 추론 속도 (ResNet18 기반)

## 🛠️ 기술 스택
- **Python 3.8+** | **PyTorch** | **ResNet18 (전이학습)**
- **Streamlit** (웹앱) | **PIL/OpenCV** (이미지 처리)

## 📊 데이터셋 정보
- **총 이미지 수**: 7,553장
  - 마스크 착용: 3,725장
  - 마스크 미착용: 3,828장
- **형식**: JPG
- **균형잡힌 데이터셋**으로 편향 없는 학습

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
# ResNet18 기반 전이학습 시작
python src/train.py --epochs 20 --batch_size 32
```

### 3. 예측 테스트
```bash
# 단일 이미지 예측
python src/predict.py --image external_images/test.jpg

# 배치 예측
python src/predict.py --folder external_images/
```

### 4. 웹 애플리케이션 실행
```bash
# Streamlit 웹앱 실행
streamlit run app/streamlit_app.py
```

## 📁 프로젝트 구조

```
DL/
│
├── 📂 data/                     # 원본 데이터셋
│   ├── with_mask/              # 마스크 착용 이미지 (3,725장)
│   └── without_mask/           # 마스크 미착용 이미지 (3,828장)
│
├── 📂 external_images/          # 외부 테스트 이미지
│   └── samples/                # 샘플 테스트 이미지
│
├── 📂 models/                   # 학습된 모델 저장
│   ├── best_model.pth          # 최고 성능 모델
│   └── training_logs.txt       # 학습 기록
│
├── 📂 src/                      # 핵심 로직
│   ├── train.py                # 학습 스크립트
│   ├── model.py                # ResNet18 모델 정의
│   ├── predict.py              # 예측 스크립트
│   ├── dataset.py              # 데이터 로더
│   └── utils.py                # 유틸리티 함수
│
├── 📂 app/                      # 웹 애플리케이션
│   └── streamlit_app.py        # Streamlit 메인 앱
│
├── requirements.txt             # Python 의존성
├── .gitignore                  # Git 무시 파일
└── README.md
```

## 🧠 모델 아키텍처

### ResNet18 전이학습
```python
# 사전 훈련된 ResNet18 사용
model = torchvision.models.resnet18(pretrained=True)

# 마지막 레이어를 2클래스 분류용으로 수정
model.fc = nn.Linear(512, 2)  # [마스크 착용, 미착용]
```

### 학습 전략
- **전이학습**: ImageNet 사전 훈련 가중치 활용
- **데이터 증강**: 회전, 밝기 조정, 좌우 반전
- **조기 종료**: 검증 정확도 기반 최적 모델 저장
- **학습률 스케줄러**: 성능 향상 시 학습률 감소

## 📊 성능 평가 지표

- **정확도 (Accuracy)** - 전체 예측 정확도
- **정밀도 (Precision)** - 마스크 착용 예측의 정확성
- **재현율 (Recall)** - 실제 마스크 착용자 탐지율
- **F1 Score** - 정밀도와 재현율의 조화평균
- **혼동 행렬** - 분류 결과 시각화

## 🌐 웹 애플리케이션 기능

### 실시간 마스크 감지
- 📤 이미지 업로드 (JPG, PNG 지원)
- ⚡ 즉시 분류 결과 표시
- 📊 예측 신뢰도 점수
- 📈 시각적 결과 차트

### 배치 처리
- 📁 여러 이미지 동시 처리
- 📋 결과 요약 리포트
- 💾 결과 CSV 다운로드

## 🚀 고급 기능

### 실시간 웹캠 감지 (선택사항)
```bash
# 웹캠 실시간 감지 (OpenCV 필요)
python src/realtime_detection.py
```

### 모델 성능 분석
```bash
# 모델 평가 및 시각화
python src/evaluate.py --model models/best_model.pth
```

## 🔧 커스터마이징

### 하이퍼파라미터 조정
`src/train.py`에서 다음 설정을 변경할 수 있습니다:
- 학습 에포크 수
- 배치 크기
- 학습률
- 데이터 증강 옵션

### 새로운 클래스 추가
1. 데이터셋에 새 폴더 추가
2. `src/model.py`에서 클래스 수 수정
3. 모델 재학습

## 📈 예상 성능
- **학습 정확도**: 95%+
- **검증 정확도**: 92%+
- **추론 속도**: <100ms (CPU)
- **모델 크기**: ~45MB

## 🔒 보안 및 개인정보

⚠️ **중요**: 이 프로젝트는 교육/연구 목적으로 제작되었습니다.
- 개인정보 보호를 위해 업로드된 이미지는 임시 저장됩니다
- 상용 서비스 적용 시 개인정보 처리 방침을 준수해야 합니다

## 🤝 기여하기

1. Fork the Project
2. Create Feature Branch (`git checkout -b feature/NewFeature`)
3. Commit Changes (`git commit -m 'Add NewFeature'`)
4. Push to Branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

😷 **마스크 착용으로 모두의 건강을 지켜요!** 🚀
