import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 데이터 로드
        self._load_data()
    
    def _load_data(self):
        """데이터 파일 경로와 라벨 생성"""
        classes = ['with_mask', 'without_mask']
        
        for class_idx, class_name in enumerate(classes):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"📊 총 데이터 수: {len(self.images)}개")
        print(f"   - with_mask: {self.labels.count(0)}개")
        print(f"   - without_mask: {self.labels.count(1)}개")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {img_path}")
            # 기본 이미지 생성 (에러 방지)
            image = Image.new('RGB', (224, 224), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """데이터 전처리 변환 정의"""
    # 학습용 (간단한 데이터 증강)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 검증/테스트용 (증강 없음)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=16, train_split=0.8):
    """데이터로더 생성"""
    train_transform, val_transform = get_transforms()
    
    # 전체 데이터셋 로드
    full_dataset = MaskDataset(data_dir, transform=train_transform)
    
    # 훈련/검증 분할
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # 검증 데이터는 다른 변환 적용
    val_dataset.dataset.transform = val_transform
    
    # 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0  # Windows 호환성
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"📚 훈련 데이터: {len(train_dataset)}개")
    print(f"📚 검증 데이터: {len(val_dataset)}개")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # 데이터셋 테스트
    data_dir = "data"
    if os.path.exists(data_dir):
        train_loader, val_loader = get_dataloaders(data_dir, batch_size=8)
        print("✅ 데이터로더 생성 완료!")
    else:
        print("❌ data 폴더를 찾을 수 없습니다.")
