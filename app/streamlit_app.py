import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
import sys

# 경로 설정 (src 폴더의 모듈 import를 위해)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from model import get_model
except ImportError:
    st.error("❌ 모델 모듈을 찾을 수 없습니다. src/model.py를 확인해주세요.")
    st.stop()


@st.cache_resource
def load_model():
    """모델 로드 (캐싱)"""
    model_path = 'models/best_model.pth'
    
    if not os.path.exists(model_path):
        return None
    
    try:
        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None


def preprocess_image(image):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def predict_image(model, image):
    """이미지 예측"""
    if model is None:
        return None, 0.0
    
    try:
        image_tensor = preprocess_image(image)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 클래스 이름
        class_names = ['😷 마스크 착용', '😐 마스크 미착용']
        result = class_names[predicted_class]
        
        return result, confidence
    
    except Exception as e:
        st.error(f"예측 중 오류: {e}")
        return None, 0.0


def main():
    # 페이지 설정
    st.set_page_config(
        page_title="마스크 착용 감지기",
        page_icon="😷",
        layout="centered"
    )
    
    # 제목
    st.title("😷 마스크 착용 감지기")
    st.markdown("### AI가 마스크 착용 여부를 판별해드립니다!")
    st.markdown("---")
    
    # 사이드바
    st.sidebar.header("📋 프로젝트 정보")
    st.sidebar.info("""
    **🛠️ 기술 스택:**
    - PyTorch + ResNet18
    - 전이학습 기반
    - Streamlit 웹앱
    
    **📊 성능:**
    - 빠른 추론 속도
    - 높은 정확도
    """)
    
    # 모델 로드 확인
    model = load_model()
    
    if model is None:
        st.error("❌ 학습된 모델을 찾을 수 없습니다!")
        st.info("💡 먼저 `python src/train.py`로 모델을 학습시켜주세요.")
        st.stop()
    
    # 메인 컨텐츠
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 이미지 업로드")
        uploaded_file = st.file_uploader(
            "얼굴 이미지를 업로드하세요",
            type=['jpg', 'jpeg', 'png'],
            help="JPG, JPEG, PNG 형식을 지원합니다"
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="업로드된 이미지", use_column_width=True)
            
            # 분석 버튼
            if st.button("🔍 마스크 착용 여부 분석", type="primary"):
                with st.spinner("AI가 분석 중입니다..."):
                    result, confidence = predict_image(model, image)
                    
                    if result is not None:
                        with col2:
                            st.header("📊 분석 결과")
                            
                            # 결과 표시
                            if "착용" in result:
                                st.success(f"### {result}")
                                st.balloons()  # 축하 효과
                            else:
                                st.warning(f"### {result}")
                            
                            # 신뢰도 표시
                            st.metric(
                                label="예측 신뢰도",
                                value=f"{confidence:.1%}",
                                help="AI가 이 예측에 대해 얼마나 확신하는지를 나타냅니다"
                            )
                            
                            # 프로그레스 바
                            st.progress(confidence)
                            
                            # 결과 해석
                            if confidence > 0.8:
                                st.info("🎯 매우 높은 신뢰도의 예측입니다!")
                            elif confidence > 0.6:
                                st.info("✅ 신뢰할 만한 예측입니다.")
                            else:
                                st.warning("⚠️ 예측 신뢰도가 낮습니다. 다른 이미지를 시도해보세요.")
                            
                           
    
    # 사용법 안내
    if uploaded_file is None:
        with col2:
            st.header("📖 사용법")
            st.markdown("""
            1. **이미지 업로드**: 왼쪽에서 얼굴이 나온 사진을 업로드하세요
            2. **분석 실행**: '마스크 착용 여부 분석' 버튼을 클릭하세요  
            3. **결과 확인**: AI가 분석한 결과를 확인하세요
            
            ### 💡 더 나은 결과를 위한 팁
            - 얼굴이 명확하게 보이는 이미지를 사용하세요
            - 정면에서 촬영한 사진이 좋습니다
            - 조명이 밝은 환경에서 찍은 사진을 권장합니다
            """)
            
            # 샘플 이미지 정보
            st.subheader("🖼️ 테스트용 샘플")
            st.markdown("""
            프로젝트의 `data` 폴더에 있는 샘플 이미지들을 테스트해보세요:
            - `data/with_mask/` - 마스크 착용 이미지
            - `data/without_mask/` - 마스크 미착용 이미지
            """)
    
    # 푸터
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "😷 마스크 착용 감지기 | ResNet18 + PyTorch + Streamlit"
        "</div>",
        unsafe_allow_html=True
    )
    
    # 건강 메시지
    st.markdown(
        "<div style='text-align: center; margin-top: 20px;'>"
        "💙 <b>마스크 착용으로 모두의 건강을 지켜요!</b> 💙"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
