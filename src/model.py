import torch
import torch.nn as nn
import torchvision.models as models
import ssl

# SSL 인증서 문제 해결
ssl._create_default_https_context = ssl._create_unverified_context


class MaskClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MaskClassifier, self).__init__()
        
        try:
            # 최신 방식으로 ResNet18 로드 (경고 제거)
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("✅ 사전 훈련된 ResNet18 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ 사전 훈련 모델 로드 실패: {e}")
            print("🔄 가중치 없이 모델 생성...")
            # 사전 훈련 없이 모델 생성 (백업)
            self.resnet = models.resnet18(weights=None)
        
        # 마지막 분류 레이어만 수정 (512 -> 2클래스)
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet(x)


def get_model():
    """모델 인스턴스 반환"""
    model = MaskClassifier(num_classes=2)
    return model


if __name__ == "__main__":
    # 간단한 모델 테스트
    print("🧪 모델 테스트 중...")
    
    model = get_model()
    print("✅ 모델 생성 완료!")
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 모델 파라미터:")
    print(f"   전체: {total_params:,}개")
    print(f"   학습 가능: {trainable_params:,}개")
    
    # 더미 입력으로 테스트
    dummy_input = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"🔍 모델 출력 테스트:")
    print(f"   입력 크기: {dummy_input.shape}")
    print(f"   출력 크기: {output.shape}")
    print(f"   예측 확률: {torch.softmax(output, dim=1).numpy()}")
