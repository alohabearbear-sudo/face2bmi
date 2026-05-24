import os
import gc
import sys
import traceback
import cv2
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image, ImageOps
from collections import Counter

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧臉部BMI預測系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 從 Streamlit Cloud Secrets 中安全讀取金鑰，絕對不外洩
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")

# --- 1. 設定 5-Fold 私有下載網址與本地路徑 ---
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# 你的私有草稿 Release 檔案下載網址
MODEL_URLS = {
    1: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/untagged-7acc0ff91d2d03279e46/fold1_best.pth",
    2: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/untagged-7acc0ff91d2d03279e46/fold2_best.pth",
    3: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/untagged-7acc0ff91d2d03279e46/fold3_best.pth",
    4: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/untagged-7acc0ff91d2d03279e46/fold4_best.pth",
    5: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/untagged-7acc0ff91d2d03279e46/fold5_best.pth"
}

FOLD_PATHS = [os.path.join(WEIGHTS_DIR, f"fold{i}_best.pth") for i in range(1, 6)]

# --- 安全下載函數 ---
def download_private_weights(fold_num):
    local_path = os.path.join(WEIGHTS_DIR, f"fold{fold_num}_best.pth")
    if not os.path.exists(local_path):
        with st.spinner(f"⏳ 正在安全下載加密 Fold {fold_num} 權重..."):
            headers = {}
            if GITHUB_TOKEN:
                # 帶上 Token 讓 GitHub 知道是擁有者要下載私有 Release
                headers["Authorization"] = f"token {GITHUB_TOKEN}"
            
            response = requests.get(MODEL_URLS[fold_num], headers=headers, stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"❌ Fold {fold_num} 下載失敗。狀態碼: {response.status_code}。請檢查 Secrets 中的 GITHUB_TOKEN 是否正確。")
                st.stop()

# --- 全域模型快取清單 ---
_models_ensemble = []

def get_ensemble_models():
    global _models_ensemble
    if not _models_ensemble:
        # 先確保 5 個模型都安全下載到伺服器本地
        for i in range(1, 6):
            download_private_weights(i)
            
        with st.spinner("⏳ 正在載入 5-Fold 交叉驗證模型群（僅在啟動時執行）..."):
            loaded_models = []
            for path in FOLD_PATHS:
                model = resnet18(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, 1)
                
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                loaded_models.append(model)
                
            _models_ensemble = loaded_models
    return _models_ensemble

# 影像預處理流程
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 2. 核心人形框繪製與 5-Fold 整合預測邏輯 ---
def process_face_bmi(img_np):
    if img_np is None:
        return None, "等待輸入...", 0.0, "等待輸入..."

    h, w = img_np.shape[:2]
    draw_img = img_np.copy()
    
    # 幾何參數定義：計算黃金比例人形虛線框
    cx, cy = int(w * 0.5), int(h * 0.45)
    color = (0, 255, 255)  # 科技黃
    thickness = max(2, int(w * 0.005))

    def draw_dashed_ellipse(img, center, axes, start_angle, end_angle, gap_deg=6):
        for a in range(start_angle, end_angle, gap_deg * 2):
            cv2.ellipse(img, center, axes, 0, a, min(a + gap_deg, end_angle), color, thickness)

    def draw_dashed_line(img, pt1, pt2, gap=12):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0: return
        pts = np.linspace(pt1, pt2, max(2, int(dist / gap)))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i+1].astype(int)), color, thickness)

    # 繪製對齊用人形虛線框
    head_axes = (int(w * 0.15), int(h * 0.2))
    draw_dashed_ellipse(draw_img, (cx, cy - int(h * 0.05)), head_axes, 0, 360)
    draw_dashed_line(draw_img, (cx - int(w * 0.05), cy + int(h * 0.15)), (cx - int(w * 0.05), cy + int(h * 0.2)), gap=8)
    draw_dashed_line(draw_img, (cx + int(w * 0.05), cy + int(h * 0.15)), (cx + int(w * 0.05), cy + int(h * 0.2)), gap=8)
    
    shoulder_y = cy + int(h * 0.2)
    draw_dashed_ellipse(draw_img, (cx - int(w * 0.22), shoulder_y + int(h * 0.1)), (int(w * 0.18), int(h * 0.1)), 270, 360, gap_deg=4)
    draw_dashed_ellipse(draw_img, (cx + int(w * 0.22), shoulder_y + int(h * 0.1)), (int(w * 0.18), int(h * 0.1)), 180, 270, gap_deg=4)
    draw_dashed_line(draw_img, (cx - int(w * 0.22), shoulder_y + int(h * 0.1)), (cx - int(w * 0.22), h))
    draw_dashed_line(draw_img, (cx + int(w * 0.22), shoulder_y + int(h * 0.1)), (cx + int(w * 0.22), h))

    try:
        # 1. 取得快取的 5 個模型
        models = get_ensemble_models()
        
        # 2. 影像預處理轉換
        pil_img = Image.fromarray(img_np)
        img_tensor = img_transforms(pil_img).unsqueeze(0)
        
        # 3. 5 個模型協同預測
        bmi_outputs = []
        gender_votes = []
        
        with torch.no_grad():
            for model in models:
                output = model(img_tensor)
                fold_bmi = output.item()
                bmi_outputs.append(fold_bmi)
                
                # 每個 Fold 根據自己的 BMI 預測結果投出性別一票
                fold_gender = "Male (男性)" if fold_bmi > 24.2 else "Female (女性)"
                gender_votes.append(fold_gender)
        
        # ✨ 核心邏輯 A：BMI 採取 5 個 Fold 的數學平均值
        bmi_val = float(np.mean(bmi_outputs))
        
        # ✨ 核心邏輯 B：性別採取 5 個 Fold 的投票多數決 (Majority Vote)
        vote_counts = Counter(gender_votes)
        gender_res = vote_counts.most_common(1)[0][0] # 找出得票數最多的性別
            
        # 4. 體態評估狀態
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
st.markdown("# 🧑‍⚕️ AI 臉部即時 BMI & 性別估算系統 by Jimmy Chen")
st.markdown("### 🎯 採用 5-Fold 交叉驗證 Ensemble 統合架構")

input_mode = st.radio("👉 請選擇輸入方式：", ["📤 上傳本機照片", "📸 開啟鏡頭拍照"], horizontal=True)

target_image = None

if input_mode == "📤 上傳本機照片":
    target_image = st.file_uploader(
        "選擇本機相簿中的正臉半身照片",
        type=["jpg", "jpeg", "png"],
        key="bmi_uploader_v5"
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

    try:
        fixed_img = ImageOps.exif_transpose(raw_img).convert('RGB')
    except Exception:
        fixed_img = raw_img.convert('RGB')

    if fixed_img.width > 1024:
        fixed_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    img_np = np.array(fixed_img)

    with st.spinner("🔍 5-Fold AI 正在綜合提取特徵與體態評估..."):
        res_draw, res_gender, res_bmi, res_status = process_face_bmi(img_np)

    with col_left:
        st.subheader("1. 臉部與對齊標記確認")
        if res_draw is not None:
            st.image(res_draw, use_container_width=True)

    with col_right:
        st.subheader("2. AI 綜合分析結果")
        
        st.subheader("📊 多數決預估性別")
        st.info(f"**{res_gender}**")
        
        st.subheader("🩺 體態評估狀態")
        st.info(f"**{res_status}**")
        
        st.subheader("🎯 5-Fold 平均 BMI 值")
        st.metric(label="Ensemble Average BMI", value=f"{res_bmi:.2f}" if res_bmi > 0 else "0.00")
else:
    gc.collect()
    with col_left:
        st.info("💡 請上傳照片或點擊上方按鈕開啟鏡頭拍照，系統將自動啟動 5-Fold AI 交叉預估。")
    with col_right:
        st.text_input("📊 多數決預估性別", value="等待輸入...", disabled=True, key="dis_gender_v5")
        st.text_input("🩺 體態評估狀態", value="等待輸入...", disabled=True, key="dis_status_v5")
        st.subheader("🎯 5-Fold 平均 BMI 值")
        st.metric(label="Ensemble Average BMI", value="0.00")

st.markdown("---")
st.markdown("<center>Developed by Jimmy Chen | 2026 Medical AI Track Edition</center>", unsafe_allow_html=True)
