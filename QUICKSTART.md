# 🚀 마스크 착용 감지기 - 빠른 시작 가이드

## 📋 준비사항
1. data 폴더에 이미지들이 준비되어 있어야 합니다
2. Python 가상환경이 활성화되어 있어야 합니다

## ⚡ 빠른 실행 순서

### 1단계: 환경 설정
```bash
# 가상환경 활성화
.venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2단계: 폴더 생성 및 확인
```bash
# 필요한 폴더들 자동 생성
python src/utils.py
```

### 3단계: 모델 학습 (약 5-10분)
```bash
# 빠른 학습 (3 에포크)
python src/train.py
```

### 4단계: 예측 테스트
```bash
# 명령어로 테스트
python src/predict.py
```

### 5단계: 웹앱 실행
```bash
# 브라우저에서 확인
streamlit run app/streamlit_app.py
```

## 🎯 교육생 제출용 특징
- **빠른 학습**: 3 에포크로 설정 (약 5-10분)
- **간단한 구조**: 복잡한 기능 제거
- **실용적**: 바로 테스트 가능한 웹앱
- **완성도**: 기본 기능 모두 포함

## 📁 최종 제출 파일들
- `src/model.py` - ResNet18 기반 모델
- `src/dataset.py` - 데이터 로더
- `src/train.py` - 학습 스크립트  
- `src/predict.py` - 예측 스크립트
- `app/streamlit_app.py` - 웹 애플리케이션
- `models/best_model.pth` - 학습된 모델

## 💡 문제 해결
1. **모델 파일 없음**: `python src/train.py` 먼저 실행
2. **이미지 없음**: data 폴더에 이미지 추가
3. **패키지 오류**: `pip install -r requirements.txt` 재실행
