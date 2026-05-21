import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import cv2
from PIL import Image
import numpy as np
import requests
import os

# 設定網頁標題
st.set_page_config(page_title="Face to BMI 偵測系統", layout="centered")
st.title("🧑‍⚕️ Face to BMI 智能預測系統")
st.write("請拍攝或上傳一張**正臉半身照**，系統將透過 AI 模型的臉部特徵估算 BMI。")

# ==========================================
# 1. 模型載入設定 (請根據你訓練時的模型架構修改)
# ==========================================
MODEL_URL = "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold1_latest.pth"
MODEL_PATH = "fold1_latest.pth"

@st.cache_resource
def load_bmi_model():
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1) 
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("首次啟動，正在從 GitHub 下載模型權重..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
                
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

try:
    bmi_model = load_bmi_model()
except Exception as e:
    st.error(f"模型載入失敗，請檢查模型架構是否與 .pth 吻合。錯誤訊息: {e}")

img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_bmi(image):
    img_t = img_transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = bmi_model(img_t)
        bmi_val = output.item()
    return bmi_val

# ==========================================
# 2. 視訊串流與動態「人形」虛線框
# ==========================================
class FaceBoxProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def draw_dashed_arc(self, img, center, axes, angle, start_angle, end_angle, color, thickness, dash_length=10, space_length=10):
        """用來繪製虛線橢圓或弧線的輔助函數"""
        # 生成完整的弧線點
        points = cv2.ellipse2Poly(center, axes, angle, start_angle, end_angle, 1)
        
        # 分段繪製以模擬虛線效果
        drawing = True
        current_len = 0
        for i in range(len(points) - 1):
            if drawing:
                cv2.line(img, tuple(points[i]), tuple(points[i+1]), color, thickness)
                current_len += np.linalg.norm(points[i+1] - points[i])
                if current_len >= dash_length:
                    drawing = False
                    current_len = 0
            else:
                current_len += np.linalg.norm(points[i+1] - points[i])
                if current_len >= space_length:
                    drawing = True
                    current_len = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 儲存未畫線的原始影格供拍照預測使用（確保預測不被黃線干擾）
        self.latest_frame = img.copy()

        color = (0, 255, 255)  # 黃色
        thickness = 2

        # 1. 定義頭部 (橢圓) 的中心與半徑
        head_center = (int(w * 0.5), int(h * 0.35))
        head_axes = (int(w * 0.15), int(h * 0.18))  # 寬半徑, 高半徑
        
        # 繪製頭部虛線橢圓
        self.draw_dashed_arc(img, head_center, head_axes, 0, 0, 360, color, thickness)

        # 2. 定義左肩與右肩 (利用圓弧繪製下垂的雙肩線條)
        # 左肩弧線
        left_shoulder_center = (int(w * 0.25), int(h * 0.85))
        left_shoulder_axes = (int(w * 0.25), int(h * 0.3))
        self.draw_dashed_arc(img, left_shoulder_center, left_shoulder_axes, 0, 270, 360, color, thickness)

        # 右肩弧線
        right_shoulder_center = (int(w * 0.75), int(h * 0.85))
        right_shoulder_axes = (int(w * 0.25), int(h * 0.3))
        self.draw_dashed_arc(img, right_shoulder_center, right_shoulder_axes, 0, 180, 270, color, thickness)
        
        # 3. 繪製身體兩側垂直向下延伸的虛線
        y_start = int(h * 0.85)
        for y in range(y_start, h, 20):
            cv2.line(img, (int(w * 0.25), y), (int(w * 0.25), min(y + 10, h)), color, thickness)
            cv2.line(img, (int(w * 0.75), y), (int(w * 0.75), min(y + 10, h)), color, thickness)

        # 提示文字
        cv2.putText(img, "Align Head & Shoulders with Contour", (int(w * 0.2), int(h * 0.1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. UI 介面設計
# ==========================================
app_mode = st.sidebar.selectbox("請選擇輸入方式", ["即時動態拍照", "上傳本機照片"])

if app_mode == "即時動態拍照":
    st.subheader("📷 鏡頭動態捕捉")
    st.info("請將頭部與雙肩對齊畫面中的黃色人形虛線框。")
    
    ctx = webrtc_streamer(
        key="face2bmi-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceBoxProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        if st.button("📸 擷取畫面並預測"):
            raw_img = ctx.video_processor.latest_frame
            if raw_img is not None:
                raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(raw_rgb)
                
                st.image(pil_img, caption="已擷取的照片 (已自動去除提示框)", use_container_width=True)
                
                with st.spinner("AI 計算中..."):
                    bmi_result = predict_bmi(pil_img)
                    
                st.success(f"### 🎯 預測 BMI 值: **{bmi_result:.2f}**")
                
                if bmi_result < 18.5:
                    st.warning("評估結果：體重過輕。記得均衡飲食吸收營養喔！")
                elif 18.5 <= bmi_result < 24:
                    st.success("評估結果：健康體重！太棒了，請繼續保持。")
                elif 24 <= bmi_result < 27:
                    st.warning("評估結果：過重。注意日常作息與飲食調整。")
                else:
                    st.error("評估結果：肥胖。建議搭配運動與專業醫療諮詢。")
            else:
                st.error("視訊暫存中尚未捕捉到影格，請稍候片刻並重試。")

elif app_mode == "上傳本機照片":
    st.subheader("📤 上傳半身照片")
    uploaded_file = st.file_uploader("請選擇一張正臉半身照片 (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="已上傳的照片", use_container_width=True)
        
        if st.button("🚀 開始分析 BMI"):
            with st.spinner("AI 進行特徵分析中..."):
                bmi_result = predict_bmi(image)
            
            st.success(f"### 🎯 預測 BMI 值: **{bmi_result:.2f}**")
            if bmi_result < 18.5:
                st.warning("評估結果：體重過輕")
            elif 18.5 <= bmi_result < 24:
                st.success("評估結果：標準體態")
            elif 24 <= bmi_result < 27:
                st.warning("評估結果：輕度過重")
            else:
                st.error("評估結果：肥胖體態")
