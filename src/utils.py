import os


def create_sample_folders():
    """샘플 폴더 구조 생성"""
    folders = [
        'data/with_mask',
        'data/without_mask', 
        'external_images/samples',
        'models'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"📁 폴더 생성: {folder}")


def check_data():
    """데이터 현황 확인"""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print("❌ data 폴더가 없습니다.")
        return False
    
    with_mask_dir = os.path.join(data_dir, "with_mask")
    without_mask_dir = os.path.join(data_dir, "without_mask")
    
    with_mask_count = 0
    without_mask_count = 0
    
    if os.path.exists(with_mask_dir):
        with_mask_count = len([f for f in os.listdir(with_mask_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if os.path.exists(without_mask_dir):
        without_mask_count = len([f for f in os.listdir(without_mask_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"📊 데이터 현황:")
    print(f"   마스크 착용: {with_mask_count}개")
    print(f"   마스크 미착용: {without_mask_count}개")
    print(f"   총 데이터: {with_mask_count + without_mask_count}개")
    
    return with_mask_count > 0 and without_mask_count > 0


def print_usage():
    """사용법 출력"""
    print("🚀 마스크 착용 감지기 - 사용법")
    print("=" * 40)
    print("1. 학습 실행:")
    print("   python src/train.py")
    print()
    print("2. 예측 테스트:")
    print("   python src/predict.py")
    print()
    print("3. 웹앱 실행:")
    print("   streamlit run app/streamlit_app.py")
    print()
    print("💡 팁:")
    print("   - data 폴더에 이미지를 넣고 학습하세요")
    print("   - 빠른 테스트를 위해 에포크는 3으로 설정됨")
    print("   - GPU가 있으면 더 빠르게 학습됩니다")


if __name__ == "__main__":
    print("🛠️ 마스크 착용 감지기 - 유틸리티")
    print("=" * 40)
    
    # 폴더 생성
    create_sample_folders()
    print()
    
    # 데이터 확인
    check_data()
    print()
    
    # 사용법 출력
    print_usage()
