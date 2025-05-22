import streamlit as st
import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import os
import urllib.request

st.set_page_config(
    page_title="이게뭘까",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "model/classifier.pt"
CLASSES_PATH = "model/classes.txt"
HF_URL = "https://huggingface.co/diplemong/brainrot-classifier/resolve/main/classifier.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 파일 자동 다운로드
if not os.path.exists(MODEL_PATH):
    os.makedirs("model", exist_ok=True)
    with st.spinner("모델이 없어서 다운로드 중입니다..."):
        urllib.request.urlretrieve(HF_URL, MODEL_PATH)
    st.success("모델 다운로드 완료!")

# 클래스 이름 로드
with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f]

# 모델 구조 설정 및 가중치 로드
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(image):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)[0]
    return probs.cpu()

# UI 시작
st.markdown("""
    <div style='text-align:center'>
        <h1 style='color:#FF4B4B;'>나와 닮은 브레인롯 캐릭터는?</h1>
        <p style='color:gray;'>AI가 당신의 얼굴을 분석해 닮은 브레인롯 캐릭터를 찾아내 줍니다</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 이미지 업로드")
uploaded_file = st.file_uploader("이미지를 업로드하세요!", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    probs = predict_image(image)
    top_idx = torch.argmax(probs).item()

    class_name_map = {
        "brrbrr": "브르르 브르르 파타핌",
        "TralaleroTralala": "트랄랄렐로 트랄랄라",
        "lirililarila": "리릴리 라릴라",
        "tung9sahur": "퉁 퉁 퉁 퉁 퉁 퉁 퉁 퉁 퉁 사후르",
        "ChimpanziniBananini": "침판치니 바나니니",
        "BombardiroCrocodilo": "봄바르딜로 크로코틸로"
    }

    raw_top_class = class_names[top_idx]
    top_class = class_name_map.get(raw_top_class, raw_top_class)
    top_prob = probs[top_idx].item()

    st.markdown("---")
    st.markdown(f"""
        <div style='padding:1em; border-radius:12px; background:#f9f9f9; text-align:center;'>
            <h2 style='color:#2ecc71;'> 당신은 <span style='color:#FF8C00;'>{top_prob*100:.2f}%</span> 비율로</h2>
            <h1 style='color:#e74c3c'>{top_class}</h1>
        </div>
    """, unsafe_allow_html=True)

    sorted_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
    labels = [class_names[i] for i, _ in sorted_probs]
    values = [float(p) for _, p in sorted_probs]

    for name, val in zip(labels, values):
        st.progress(val, text=f"{name} ({val*100:.2f}%)")
