import os
import gc
import sys
import traceback
import requests
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image, ImageOps

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧臉部BMI預測系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. 設定與路記 ---
MODEL_URL = "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold1_latest.pth"
MODEL_PATH = "fold1_latest.pth"

# --- 全域模型快取 ---
_model = None

def get_model():
    global _model
    if _model is None:
        with st.spinner("⏳ 正在安全載入 AI 預測模型..."):
            model = resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 1) 
            
            # 如果本地無模型則自動下載
            if not os.path.exists(MODEL_PATH):
                response = requests.get(MODEL_URL)
                with open(MODEL_PATH, "wb") as f:
                    f.write(response.content)
            
            state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            _model = model
    return _model

# 影像預處理流程
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. 核心人形框繪製與預測邏輯 ---
def process_face_bmi(img_np):
    if img_np is None:
        return None, "等待輸入...", 0.0, "等待輸入..."

    h, w = img_np.shape[:2]
    draw_img = img_np.copy()
    
    # 幾何參數定義：精準計算完美比例人形虛線框
    cx, cy = int(w * 0.5), int(h * 0.45)
    color = (0, 255, 255)  # 科技黃
    thickness = max(2, int(w * 0.005)) # 隨影像大小調整線條粗細

    # 模擬虛線橢圓輔助函數
    def draw_dashed_ellipse(img, center, axes, start_angle, end_angle, gap_deg=6):
        for a in range(start_angle, end_angle, gap_deg * 2):
            cv2.ellipse(img, center, axes, 0, a, min(a + gap_deg, end_angle), color, thickness)

    # 模擬虛線直線輔助函數
    def draw_dashed_line(img, pt1, pt2, gap=12):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0: return
        pts = np.linspace(pt1, pt2, max(2, int(dist / gap)))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i+1].astype(int)), color, thickness)

    # 1. 繪製人形虚線框 (讓使用者對齊)
    # 頭部 (蛋形)
    head_axes = (int(w * 0.15), int(h * 0.2))
    draw_dashed_ellipse(draw_img, (cx, cy - int(h * 0.05)), head_axes, 0, 360)
    # 脖子
    draw_dashed_line(draw_img, (cx - int(w * 0.05), cy + int(h * 0.15)), (cx - int(w * 0.05), cy + int(h * 0.2)), gap=8)
    draw_dashed_line(draw_img, (cx + int(w * 0.05), cy + int(h * 0.15)), (cx + int(w * 0.05), cy + int(h * 0.2)), gap=8)
    # 雙肩
    shoulder_y = cy + int(h * 0.2)
    draw_dashed_ellipse(draw_img, (cx - int(w * 0.22), shoulder_y + int(h * 0.1)), (int(w * 0.18), int(h * 0.1)), 270, 360, gap_deg=4)
    draw_dashed_ellipse(draw_img, (cx + int(w * 0.22), shoulder_y + int(h * 0.1)), (int(w * 0.18), int(h * 0.1)), 180, 270, gap_deg=4)
    # 身體兩側下切
    draw_dashed_line(draw_img, (cx - int(w * 0.22), shoulder_y + int(h * 0.1)), (cx - int(w * 0.22), h))
    draw_dashed_line(draw_img, (cx + int(w * 0.22), shoulder_y + int(h * 0.1)), (cx + int(w * 0.22), h))

    try:
        # 模型預測
        model = get_model()
        pil_img = Image.fromarray(img_np)
        img_tensor = img_transforms(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            bmi_val = output.item()
            
        # 性別與健康狀況評估邏輯
        gender_res = "Male (男性)" if bmi_val > 24.2 else "Female (女性)"
        
        if bmi_val < 18.5:
            status_res = "🔵 體重過輕"
        elif 18.5 <= bmi_val < 24:
            status_res = "🟢 健康體態"
        elif 24 <= bmi_val < 27:
            status_res = "🟡 輕度過重"
        else:
            status_res = "🔴 肥胖體態"

    except Exception as e:
        bmi_val = 0.0
        gender_res = "Error"
        status_res = f"❌ 辨識異常: {str(e)}"
        print(traceback.format_exc())
    finally:
        gc.collect()

    return draw_img, gender_res, bmi_val, status_res

# --- 3. 前端 CSS 風格 ---
st.markdown("""
<style>
    .stMarkdown h1 { color: #2E7D32; text-align: center; font-weight: bold; }
    .stMarkdown h3 { text-align: center; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- 4. 建立 UI 介面 ---
st.markdown("# 🧑‍⚕️ AI 顏值體重計 by Jimmy Chen")
st.markdown("### 🎯 採用 ResNet 骨幹任務迴歸架構")

# 提供兩種模式供切換：上傳相簿照片 或 現場拍照
input_mode = st.radio("👉 請選擇輸入方式：", ["📤 上傳本機照片", "📸 開啟鏡頭拍照"], horizontal=True)

target_image = None

if input_mode == "📤 上傳本機照片":
    target_image = st.file_uploader(
        "選擇本機相簿中的正臉半身照片",
        type=["jpg", "jpeg", "png"],
        key="bmi_uploader"
    )
else:
    target_image = st.camera_input("請將正臉與肩膀對齊畫面中央進行拍攝")

st.write("---")

# --- 5. 畫面渲染與雙欄位對齊 ---
col_left, col_right = st.columns([3, 2])

if target_image is not None:
    try:
        target_image.seek(0)
        raw_img = Image.open(target_image)
        raw_img.load()
    except Exception as e:
        st.error(f"圖片讀取失敗: {e}")
        st.stop()

    # 校正照片旋轉角度
    try:
        fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    except Exception:
        fixed_img = raw_img.convert('RGB')

    # 縮放過大影像提升推論效率
    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    img_np = np.array(fixed_img)

    with st.spinner("🔍 AI 正在進行特徵提取與體態評估..."):
        res_draw, res_gender, res_bmi, res_status = process_face_bmi(img_np)

    with col_left:
        st.subheader("1. 臉部與對齊標記確認")
        if res_draw is not None:
            # 渲染帶有人形黃色虛線框的確認影像
            st.image(res_draw, use_container_width=True)

    with col_right:
        st.subheader("2. AI 即時分析結果")
        
        st.subheader("📊 預估性別")
        st.info(f"**{res_gender}**")
        
        st.subheader("🩺 體態評估狀態")
        st.info(f"**{res_status}**")
        
        st.subheader("🎯 預測 BMI 值")
        st.metric(label="Calculated BMI", value=f"{res_bmi:.2f}" if res_bmi > 0 else "0.00")
else:
    gc.collect()
    with col_left:
        st.info("💡 請上傳照片或點擊上方「允許」開啟鏡頭拍照，系統將自動啟動 AI 特徵估算。")
    with col_right:
        st.text_input("📊 預估性別", value="等待輸入...", disabled=True, key="dis_gender")
        st.text_input("🩺 體態評估狀態", value="等待輸入...", disabled=True, key="dis_status")
        st.subheader("🎯 預測 BMI 值")
        st.metric(label="Calculated BMI", value="0.00")

st.markdown("---")
st.markdown("<center>Developed by Jimmy Chen | 2026 Medical AI Track Edition</center>", unsafe_allow_html=True)
