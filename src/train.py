import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import matplotlib.pyplot as plt
from model import get_model
from dataset import get_dataloaders


def train_model(data_dir="data", epochs=5, batch_size=16, lr=0.001):
    """간단하고 빠른 모델 학습 + 시각화"""
    
    print("🚀 마스크 착용 감지기 학습 시작!")
    print("=" * 50)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 사용 디바이스: {device}")
    
    # 데이터 로더 생성
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    
    # 모델 생성
    model = get_model()
    model = model.to(device)
    
    # 손실 함수와 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 학습 기록 (시각화용)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_acc = 0.0
    
    print(f"📚 배치 크기: {batch_size}, 학습률: {lr}, 에포크: {epochs}")
    print("=" * 50)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # 훈련 단계
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"📖 Epoch {epoch+1}/{epochs} - 훈련 중...")
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 순전파
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 통계
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # 진행상황 출력 (10배치마다)
            if batch_idx % 10 == 0:
                print(f"   배치 {batch_idx}/{len(train_loader)}, 손실: {loss.item():.4f}")
        
        # 훈련 결과
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 검증 단계
        model.eval()
        val_correct = 0
        val_total = 0
        
        print(f"🔍 검증 중...")
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # 결과 저장 (시각화용)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 최고 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"💾 새로운 최고 모델 저장! (정확도: {val_acc:.2f}%)")
        
        # 에포크 결과 출력
        epoch_time = time.time() - start_time
        print(f"✅ Epoch {epoch+1} 완료 ({epoch_time:.1f}초)")
        print(f"   훈련 손실: {train_loss:.4f}, 훈련 정확도: {train_acc:.2f}%")
        print(f"   검증 정확도: {val_acc:.2f}%")
        print("-" * 30)
    
    print("🎉 학습 완료!")
    print(f"🏆 최고 검증 정확도: {best_val_acc:.2f}%")
    print(f"📁 모델 저장 위치: models/best_model.pth")
    
    # 📊 학습 곡선 시각화 생성
    create_training_plots(train_losses, train_accuracies, val_accuracies, epochs)
    
    return model, train_losses, train_accuracies, val_accuracies


def create_training_plots(train_losses, train_accuracies, val_accuracies, epochs):
    """학습 곡선 시각화 생성"""
    
    print("📈 학습 곡선 시각화 생성 중...")
    os.makedirs('results', exist_ok=True)
    
    # 한글 폰트 설정 (Windows)
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 그래프 생성 (2x1 레이아웃)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    epochs_range = range(1, epochs + 1)
    
    # 1. 손실 그래프
    ax1.plot(epochs_range, train_losses, 'b-', label='훈련 손실', linewidth=2)
    ax1.set_title('🔄 모델 학습 손실 변화', fontsize=14, fontweight='bold')
    ax1.set_xlabel('에포크')
    ax1.set_ylabel('손실 (Loss)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 정확도 그래프
    ax2.plot(epochs_range, train_accuracies, 'g-', label='훈련 정확도', linewidth=2)
    ax2.plot(epochs_range, val_accuracies, 'r-', label='검증 정확도', linewidth=2)
    ax2.set_title('📊 모델 정확도 변화', fontsize=14, fontweight='bold')
    ax2.set_xlabel('에포크')
    ax2.set_ylabel('정확도 (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 최종 결과 텍스트 추가
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    ax2.text(0.02, 0.98, f'최종 훈련 정확도: {final_train_acc:.1f}%\n최종 검증 정확도: {final_val_acc:.1f}%', 
             transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # 저장
    save_path = 'results/training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 학습 곡선 저장 완료: {save_path}")
    return save_path


def main():
    """메인 실행 함수"""
    # 빠른 학습을 위한 설정
    EPOCHS = 3  # 에포크 수 적게
    BATCH_SIZE = 32  # 배치 크기 크게 (빠른 학습)
    LEARNING_RATE = 0.001
    
    print("😷 마스크 착용 감지기 - 빠른 학습 버전")
    print(f"⚡ 설정: {EPOCHS} 에포크, 배치 {BATCH_SIZE}")
    
    # 데이터 폴더 확인
    if not os.path.exists("data"):
        print("❌ 오류: 'data' 폴더를 찾을 수 없습니다!")
        print("💡 data/with_mask/ 와 data/without_mask/ 폴더를 생성하고 이미지를 넣어주세요.")
        return
    
    try:
        # 모델 학습
        model, losses, train_accs, val_accs = train_model(
            data_dir="data",
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE
        )
        
        # 간단한 결과 요약
        print("\n📊 학습 결과 요약:")
        print(f"   최종 훈련 손실: {losses[-1]:.4f}")
        print(f"   최종 훈련 정확도: {train_accs[-1]:.2f}%")
        print(f"   최종 검증 정확도: {val_accs[-1]:.2f}%")
        print(f"📈 시각화 결과는 results/ 폴더에서 확인하세요!")
        
    except Exception as e:
        print(f"❌ 학습 중 오류 발생: {e}")
        print("💡 데이터 경로와 이미지 파일을 확인해주세요.")


if __name__ == "__main__":
    main()
