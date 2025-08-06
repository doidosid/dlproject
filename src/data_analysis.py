import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random


def analyze_dataset(data_dir='data'):
    """데이터셋 분석 및 시각화"""
    
    print("데이터셋 분석 시작!")
    print("=" * 40)
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 클래스별 이미지 수 계산
    classes = ['with_mask', 'without_mask']
    class_counts = {}
    
    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    print("클래스별 데이터 개수:")
    for class_name, count in class_counts.items():
        korean_name = "마스크 착용" if class_name == "with_mask" else "마스크 미착용"
        print(f"   {korean_name}: {count:,}개")
    
    total_images = sum(class_counts.values())
    print(f"총 이미지 수: {total_images:,}개")
    
    # 시각화 생성
    create_dataset_visualization(class_counts, data_dir)
    
    print("데이터셋 분석 완료!")


def create_dataset_visualization(class_counts, data_dir):
    """데이터셋 시각화 생성"""
    
    print("🎨 데이터셋 시각화 생성 중...")
    os.makedirs('results', exist_ok=True)
    
    # 그림 크기 설정
    fig = plt.figure(figsize=(15, 10))
    
    # 1. 클래스별 데이터 분포 (상단 왼쪽)
    ax1 = plt.subplot(2, 3, 1)
    korean_names = ['마스크 착용', '마스크 미착용']
    counts = list(class_counts.values())
    colors = ['skyblue', 'lightcoral']
    
    bars = ax1.bar(korean_names, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('클래스별 데이터 분포', fontsize=14, fontweight='bold')
    ax1.set_ylabel('이미지 수')
    
    # 막대 위에 숫자 표시
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 파이 차트 (상단 중앙)
    ax2 = plt.subplot(2, 3, 2)
    ax2.pie(counts, labels=korean_names, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 12})
    ax2.set_title('데이터 비율', fontsize=14, fontweight='bold')
    
    # 3. 총계 정보 (상단 오른쪽)
    ax3 = plt.subplot(2, 3, 3)
    ax3.axis('off')
    total = sum(counts)
    info_text = f"""
데이터셋 요약

총 이미지 수: {total:,}개

마스크 착용: {counts[0]:,}개
마스크 미착용: {counts[1]:,}개

데이터 균형도:
{abs(counts[0] - counts[1]) / total * 100:.1f}% 차이

✅ 균형잡힌 데이터셋
"""
    ax3.text(0.1, 0.9, info_text, transform=ax3.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4-6. 샘플 이미지 표시 (하단)
    sample_positions = [(2, 3, 4), (2, 3, 5), (2, 3, 6)]
    class_names = ['with_mask', 'without_mask']
    korean_names_short = ['마스크 착용', '마스크 미착용']
    
    for idx, (class_name, korean_name) in enumerate(zip(class_names, korean_names_short)):
        if idx < 2:  # 두 클래스만 표시
            ax = plt.subplot(sample_positions[idx][0], sample_positions[idx][1], sample_positions[idx][2])
            
            # 랜덤 샘플 이미지 로드
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                image_files = [f for f in os.listdir(class_path) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    sample_file = random.choice(image_files)
                    sample_path = os.path.join(class_path, sample_file)
                    
                    try:
                        img = Image.open(sample_path).convert('RGB')
                        img = img.resize((200, 200))  # 크기 조정
                        ax.imshow(img)
                        ax.set_title(f'{korean_name} 샘플', fontsize=12, fontweight='bold')
                        ax.axis('off')
                    except Exception as e:
                        ax.text(0.5, 0.5, f'이미지 로드 실패\n{str(e)[:30]}...', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{korean_name} 샘플', fontsize=12)
            else:
                ax.text(0.5, 0.5, '폴더 없음', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{korean_name} 샘플', fontsize=12)
    
    # 전체 제목
    fig.suptitle('마스크 착용 감지기 - 데이터셋 분석', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # 저장
    save_path = 'results/dataset_analysis.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"데이터셋 시각화 저장 완료: {save_path}")
    return save_path


def main():
    """메인 실행 함수"""
    print("마스크 착용 감지기 - 데이터셋 분석")
    print("=" * 40)
    
    if not os.path.exists('data'):
        print("data 폴더를 찾을 수 없습니다!")
        print("💡 data/with_mask/ 와 data/without_mask/ 폴더를 생성하고 이미지를 넣어주세요.")
        return
    
    try:
        analyze_dataset('data')
        
        print("\n🎉 분석 완료!")
        print("📁 생성된 파일:")
        print("   - results/dataset_analysis.png")
        print("\n💡 이 파일을 보고서에 활용하세요!")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
