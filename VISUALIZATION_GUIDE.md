# 🚀 보고서용 시각화 - 빠른 실행 가이드

## 📊 생성되는 시각화 파일들

### 1단계: 데이터 분석 (2분)
```bash
python src/data_analysis.py
```
**결과**: `results/dataset_analysis.png`
- 클래스별 데이터 분포
- 파이 차트
- 샘플 이미지

### 2단계: 모델 학습 + 학습곡선 (10-15분)
```bash
python src/train.py
```
**결과**: `results/training_curves.png`
- 손실 변화 그래프
- 정확도 변화 그래프
- 최종 성능 수치

### 3단계: 모델 평가 + 성능 시각화 (3분)
```bash
python src/evaluate.py
```
**결과**:
- `results/confusion_matrix.png` - 혼동행렬
- `results/classification_report.txt` - 성능 지표
- `results/prediction_samples.png` - 예측 샘플

## 🎯 보고서 구성 제안

### 📄 보고서 순서
1. **서론**: 프로젝트 배경 및 목표
2. **데이터셋**: `dataset_analysis.png` 활용
3. **방법론**: 모델 구조 및 학습 전략 설명
4. **학습 과정**: `training_curves.png` 활용
5. **성능 평가**: `confusion_matrix.png` + `classification_report.txt` 활용
6. **결과 분석**: `prediction_samples.png` 활용
7. **결론**: 성과 및 한계점

### 📊 각 시각화별 설명 포인트

#### 1. 데이터셋 분석
- 총 7,553개의 균형잡힌 데이터셋
- 마스크 착용/미착용 클래스 분포
- 실제 샘플 이미지 확인

#### 2. 학습 곡선  
- 손실이 점진적으로 감소
- 과적합 여부 확인 (train vs validation)
- 최종 성능 수치

#### 3. 혼동행렬
- 각 클래스별 예측 정확도
- 오분류 패턴 분석
- 전체 정확도 확인

#### 4. 예측 샘플
- 모델의 실제 예측 결과
- 올바른 예측과 오류 사례
- 신뢰도 점수 분석

## ⚡ 전체 실행 (한 번에)

### Windows (PowerShell)
```powershell
# 가상환경 활성화
.venv\Scripts\activate

# 전체 시각화 생성
python src/data_analysis.py; python src/train.py; python src/evaluate.py

# 결과 확인
ls results/
```

## 📁 최종 결과물
```
results/
├── 📊 dataset_analysis.png     # 데이터셋 분석
├── 📈 training_curves.png      # 학습 곡선
├── 🎯 confusion_matrix.png     # 혼동행렬
├── 📋 classification_report.txt # 성능 지표
└── 🔍 prediction_samples.png   # 예측 샘플
```

## 💡 보고서 작성 팁

### 📊 수치 활용
- 정확도, F1-Score 등 구체적 수치 인용
- 학습 시간, 에포크 수 등 실험 설정 명시
- 데이터 크기, 모델 파라미터 수 등 기술적 정보

### 🖼️ 시각화 활용
- 각 그래프의 의미와 해석 설명
- 성능 개선 방향이나 한계점 분석
- 실제 예측 결과로 모델 성능 입증

### 📝 교육생다운 표현
- "ResNet18 전이학습을 활용하여..."
- "데이터 증강을 통해 모델 일반화 성능 향상..."
- "혼동행렬 분석 결과 XX% 정확도 달성..."

## 🎯 예상 소요 시간
- **데이터 분석**: 2분
- **모델 학습**: 10-15분 (3 에포크)
- **성능 평가**: 3분
- **총 소요 시간**: 약 20분

이제 바로 시작하세요! 🚀
