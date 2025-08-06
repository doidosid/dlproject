import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from model import get_model
from dataset import get_dataloaders
import os


def evaluate_model(model_path='models/best_model.pth', data_dir='data'):
    """모델 평가 및 혼동행렬 생성"""
    
    print("🔍 모델 평가 시작!")
    print("=" * 40)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("💡 먼저 python src/train.py로 모델을 학습시켜주세요.")
        return
    
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료: {model_path}")
    
    # 데이터 로더 생성 (검증용)
    _, val_loader = get_dataloaders(data_dir, batch_size=32)
    
    # 예측 수행
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("예측 수행 중...")
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 정확도 계산
    accuracy = (np.array(all_predictions) == np.array(all_labels)).mean() * 100
    print(f"전체 정확도: {accuracy:.2f}%")
    
    # 혼동행렬 생성
    create_confusion_matrix(all_labels, all_predictions)
    
    # 분류 보고서 생성
    create_classification_report(all_labels, all_predictions)
    
    # 예측 샘플 시각화
    create_prediction_samples(val_loader, model, device)
    
    print("평가 완료! results/ 폴더를 확인하세요.")


def create_confusion_matrix(true_labels, predictions):
    """혼동행렬 시각화 생성"""
    
    print("혼동행렬 생성 중...")
    os.makedirs('results', exist_ok=True)
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 혼동행렬 계산
    cm = confusion_matrix(true_labels, predictions)
    
    # 클래스 이름
    class_names = ['마스크 착용', '마스크 미착용']
    
    # 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '예측 개수'})
    
    plt.title('혼동 행렬 (Confusion Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('예측 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    # 정확도 정보 추가
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    plt.figtext(0.02, 0.02, f'전체 정확도: {accuracy:.1f}%', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    save_path = 'results/confusion_matrix.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"혼동행렬 저장 완료: {save_path}")
    return save_path


def create_classification_report(true_labels, predictions):
    """분류 보고서 생성"""
    
    print("분류 보고서 생성 중...")
    
    class_names = ['마스크 착용', '마스크 미착용']
    report = classification_report(true_labels, predictions, 
                                 target_names=class_names, 
                                 digits=3)
    
    # 텍스트 파일로 저장
    os.makedirs('results', exist_ok=True)
    save_path = 'results/classification_report.txt'
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("마스크 착용 감지기 - 성능 평가 보고서\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)
        f.write("\n\n성능 지표 설명:\n")
        f.write("- Precision (정밀도): 예측한 것 중 맞춘 비율\n")
        f.write("- Recall (재현율): 실제 정답 중 찾아낸 비율\n")
        f.write("- F1-Score: 정밀도와 재현율의 조화평균\n")
        f.write("- Support: 각 클래스의 실제 샘플 수\n")
    
    print(f"분류 보고서 저장 완료: {save_path}")
    print("주요 성능 지표:")
    print(report)
    
    return save_path


def create_prediction_samples(val_loader, model, device, num_samples=8):
    """예측 샘플 시각화"""
    
    print("🖼️ 예측 샘플 시각화 생성 중...")
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    class_names = ['마스크 착용', '마스크 미착용']
    
    # 샘플 이미지 수집
    sample_images = []
    sample_labels = []
    sample_predictions = []
    sample_probabilities = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            if len(sample_images) >= num_samples:
                break
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            for i in range(min(len(images), num_samples - len(sample_images))):
                sample_images.append(images[i].cpu())
                sample_labels.append(labels[i].item())
                sample_predictions.append(predicted[i].item())
                sample_probabilities.append(probabilities[i].cpu().numpy())
    
    # 시각화
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
    fig.suptitle('예측 결과 샘플', fontsize=16, fontweight='bold')
    
    for idx in range(num_samples):
        row = idx // cols
        col = idx % cols
        
        # 이미지 정규화 해제 및 표시
        img = sample_images[idx]
        img = img.permute(1, 2, 0)
        
        # 정규화 해제 (ImageNet 표준) - 타입 통일
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = std * img + mean
        img = torch.clamp(img, 0, 1)  # PyTorch의 clamp 사용
        
        # NumPy로 변환하여 표시
        img_np = img.detach().cpu().numpy()
        
        axes[row, col].imshow(img_np)
        axes[row, col].axis('off')
        
        # 제목 설정
        true_label = class_names[sample_labels[idx]]
        pred_label = class_names[sample_predictions[idx]]
        confidence = sample_probabilities[idx][sample_predictions[idx]] * 100
        
        # 색상 설정 (맞으면 녹색, 틀리면 빨간색)
        color = 'green' if sample_labels[idx] == sample_predictions[idx] else 'red'
        
        title = f'실제: {true_label}\n예측: {pred_label}\n신뢰도: {confidence:.1f}%'
        axes[row, col].set_title(title, fontsize=10, color=color)
    
    plt.tight_layout()
    
    # 저장
    save_path = 'results/prediction_samples.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"예측 샘플 저장 완료: {save_path}")
    return save_path


def main():
    """메인 실행 함수"""
    print("마스크 착용 감지기 - 성능 평가")
    print("=" * 40)
    
    try:
        evaluate_model()
        
        print("\n 평가 완료!")
        print("생성된 파일들:")
        print("   - results/confusion_matrix.png")
        print("   - results/classification_report.txt") 
        print("   - results/prediction_samples.png")
        print("\n 이 파일들을 보고서에 활용하세요!")
        
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        print(" 모델이 학습되었는지 확인해주세요.")


if __name__ == "__main__":
    main()
