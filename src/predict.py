import torch
from PIL import Image
import torchvision.transforms as transforms
from model import get_model
import os


def load_model(model_path='models/best_model.pth'):
    """학습된 모델 로드"""
    model = get_model()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        print(f"✅ 모델 로드 완료: {model_path}")
        return model
    else:
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("💡 먼저 python src/train.py 로 모델을 학습시켜주세요.")
        return None


def preprocess_image(image_path):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return None


def predict_single(model, image_path):
    """단일 이미지 예측"""
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        return None, 0.0
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 클래스 이름 매핑
    class_names = ['마스크 착용', '마스크 미착용']
    result = class_names[predicted_class]
    
    return result, confidence


def main():
    """메인 실행 함수"""
    print("🔍 마스크 착용 감지기 - 예측 테스트")
    print("=" * 40)
    
    # 모델 로드
    model = load_model()
    if model is None:
        return
    
    # 테스트 이미지 경로들
    test_images = [
        "external_images/samples/test_image_1.jpg",  # 실제 존재하는 파일
        "data/with_mask/with_mask_1.jpg",
        "data/without_mask/without_mask_1.jpg"
    ]
    
    print("📸 테스트 이미지 예측 결과:")
    print("-" * 40)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            result, confidence = predict_single(model, img_path)
            if result:
                print(f"📁 {img_path}")
                print(f"   🎯 예측: {result}")
                print(f"   📊 신뢰도: {confidence:.2%}")
                print()
            else:
                print(f"❌ {img_path} - 예측 실패")
        else:
            print(f"⚠️  {img_path} - 파일 없음")
    
    # 사용자 입력으로 테스트
    print("💡 직접 테스트하려면 이미지 경로를 입력하세요 (종료: q)")
    
    while True:
        user_input = input("🖼️  이미지 경로: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if os.path.exists(user_input):
            result, confidence = predict_single(model, user_input)
            if result:
                print(f"   🎯 예측: {result}")
                print(f"   📊 신뢰도: {confidence:.2%}")
            else:
                print("   ❌ 예측 실패")
        else:
            print("   ⚠️  파일을 찾을 수 없습니다.")
        
        print()


if __name__ == "__main__":
    main()
