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
# 1. 模型載入設定 (請根據你訓練時的模型架換修改)
# ==========================================
MODEL_URL = "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold1_latest.pth"
MODEL_PATH = "fold1_latest.pth"

@st.cache_resource
def load_bmi_model():
    # 這裡假設你的模型是 ResNet18 用於迴歸任務 (輸出 1 個 BMI 值)
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1) 
    
    # 下載權重
    if not os.path.exists(MODEL_PATH):
        with st.spinner("首次啟動，正在從 GitHub 下載模型權重..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
                
    # 載入權重 (相容 CPU 部署環境)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

try:
    bmi_model = load_bmi_model()
except Exception as e:
    st.error(f"模型載入失敗，請檢查模型架構是否與 .pth 吻合。錯誤訊息: {e}")

# 圖像預處理流程 (請依據訓練時的 size 與 normalize 調整)
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_bmi(image):
    """輸入 PIL Image，輸出預測的 BMI 數值"""
    img_t = img_transforms(image).unsqueeze(0)
    with torch.no_grad():
        output = bmi_model(img_t)
        bmi_val = output.item()
    return bmi_val

# ==========================================
# 2. 新版視訊串流與動態虛線框 (使用 VideoProcessorBase)
# ==========================================
class FaceBoxProcessor(VideoProcessorBase):
    def __init__(self):
        self.latest_frame = None

    def recv(self, frame):
        # 轉換為 ndarray (BGR 格式)
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # 儲存未畫線的原始影格供拍照使用
        self.latest_frame = img.copy()

        # 計算半身/臉部虛線框位置 (置中，佔寬度的 50%，高度的 70%)
        box_w, box_h = int(w * 0.5), int(h * 0.7)
        x1, y1 = int((w - box_w) / 2), int((h - box_h) / 4)
        x2, y2 = x1 + box_w, y1 + box_h

        # 繪製黃色虛線框
        color = (0, 255, 255) # 黃色
        thickness = 2
        
        # 模擬虛線
        for i in range(x1, x2, 15):
            cv2.line(img, (i, y1), (min(i + 8, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i + 8, x2), y2), color, thickness)
        for i in range(y1, y2, 15):
            cv2.line(img, (x1, i), (x1, min(i + 8, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i + 8, y2)), color, thickness)
            
        # 提示文字
        cv2.putText(img, "Align Face & Upper Body Inside Box", (x1 - 10, y1 - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        return frame.from_ndarray(img, format="bgr24")

# ==========================================
# 3. UI 介面設計
# ==========================================
app_mode = st.sidebar.selectbox("請選擇輸入方式", ["即時動態拍照", "上傳本機照片"])

if app_mode == "即時動態拍照":
    st.subheader("📷 鏡頭動態捕捉")
    st.info("請將頭部與雙肩對齊畫面中的黃色虛線框。")
    
    # 啟動 WebRTC 串流 (更新為 video_processor_factory)
    ctx = webrtc_streamer(
        key="face2bmi-stream",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=FaceBoxProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    # 拍照按鈕 (更新為 ctx.video_processor)
    if ctx.video_processor:
        if st.button("📸 擷取畫面並預測"):
            raw_img = ctx.video_processor.latest_frame
            if raw_img is not None:
                # 轉換為 RGB 與 PIL 格式
                raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(raw_rgb)
                
                # 顯示拍到的照片
                st.image(pil_img, caption="已擷取的照片 (已自動去除提示框)", use_container_width=True)
                
                # 預測
                with st.spinner("AI 計算中..."):
                    bmi_result = predict_bmi(pil_img)
                    
                # 呈現結果
                st.success(f"### 🎯 預測 BMI 值: **{bmi_result:.2f}**")
                
                # BMI 健康狀態評估 (台灣國健署標準)
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
