import os
import gc
import traceback
import cv2
import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
from PIL import Image, ImageOps

# --- 0. Streamlit 網頁基本配置 ---
st.set_page_config(
    page_title="AI智慧臉部BMI預測系統",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1. 下載設定 ---
WEIGHTS_DIR = "weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)

MODEL_URLS = {
    1: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold1_best.pth",
    2: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold2_best.pth",
    3: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold3_best.pth",
    4: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold4_best.pth",
    5: "https://github.com/alohabearbear-sudo/face2bmi/releases/download/v1/fold5_best.pth",
}
FOLD_PATHS = [os.path.join(WEIGHTS_DIR, f"fold{i}_best.pth") for i in range(1, 6)]

# 訓練時 BMI 標準化參數（meta=0 時有效）
BMI_MEAN = 24.5
BMI_STD  = 4.5


# --- 2. 正確模型架構 ---
class FaceBMIModel(nn.Module):
    """
    Backbone : EfficientNet-B3 → 1536-dim
    meta_encoder : gender scalar (0=Female / 1=Male) → 64-dim
    bmi_head     : [1536+64=1600] → 1  (標準化 BMI)
    gender_head  : [1536] → 1  (未學好，不使用)
    """
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3', pretrained=False,
            num_classes=0, global_pool='avg'
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 64),
        )
        self.bmi_head = nn.Sequential(
            nn.Linear(1600, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )
        self.gender_head = nn.Sequential(
            nn.Linear(1536, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
        )

    def forward(self, x, meta):
        feat         = self.backbone(x)
        meta_feat    = self.meta_encoder(meta)
        bmi_norm     = self.bmi_head(torch.cat([feat, meta_feat], dim=1))
        gender_logit = self.gender_head(feat)
        return bmi_norm, gender_logit


# --- 3. 下載權重 ---
def download_public_weights(fold_num):
    local_path = FOLD_PATHS[fold_num - 1]
    if not os.path.exists(local_path):
        with st.spinner(f"⏳ 正在從雲端下載 Fold {fold_num} 權重檔案..."):
            response = requests.get(MODEL_URLS[fold_num], stream=True)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                st.error(f"❌ Fold {fold_num} 下載失敗。狀態碼: {response.status_code}")
                st.stop()


# --- 4. 全域模型快取 ---
_models_ensemble = []

def get_ensemble_models():
    global _models_ensemble
    if not _models_ensemble:
        for i in range(1, 6):
            download_public_weights(i)
        with st.spinner("⏳ 載入 5-Fold EfficientNet-B3 Ensemble..."):
            loaded = []
            for path in FOLD_PATHS:
                model = FaceBMIModel()
                try:
                    sd = torch.load(path, map_location='cpu', weights_only=False)
                except Exception:
                    sd = torch.load(path, map_location='cpu')
                model.load_state_dict(sd, strict=True)
                model.eval()
                loaded.append(model)
            _models_ensemble = loaded
    return _models_ensemble


# --- 5. 影像預處理 ---
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# --- 6. 核心推理 ---
def process_face_bmi(img_np, gender_val: float):
    """
    gender_val : 0.0 = Female, 1.0 = Male（由使用者選擇）
    """
    if img_np is None:
        return None, 0.0, "等待輸入..."

    h, w = img_np.shape[:2]
    draw_img  = img_np.copy()
    cx, cy    = int(w * 0.5), int(h * 0.45)
    color     = (0, 255, 255)
    thickness = max(2, int(w * 0.005))

    def draw_dashed_ellipse(img, center, axes, start_angle, end_angle, gap_deg=6):
        for a in range(start_angle, end_angle, gap_deg * 2):
            cv2.ellipse(img, center, axes, 0, a,
                        min(a + gap_deg, end_angle), color, thickness)

    def draw_dashed_line(img, pt1, pt2, gap=12):
        dist = np.linalg.norm(np.array(pt1) - np.array(pt2))
        if dist == 0:
            return
        pts = np.linspace(pt1, pt2, max(2, int(dist / gap)))
        for i in range(0, len(pts) - 1, 2):
            cv2.line(img, tuple(pts[i].astype(int)),
                     tuple(pts[i + 1].astype(int)), color, thickness)

    head_axes  = (int(w * 0.15), int(h * 0.2))
    draw_dashed_ellipse(draw_img, (cx, cy - int(h * 0.05)), head_axes, 0, 360)
    draw_dashed_line(draw_img,
                     (cx - int(w * 0.05), cy + int(h * 0.15)),
                     (cx - int(w * 0.05), cy + int(h * 0.2)), gap=8)
    draw_dashed_line(draw_img,
                     (cx + int(w * 0.05), cy + int(h * 0.15)),
                     (cx + int(w * 0.05), cy + int(h * 0.2)), gap=8)
    shoulder_y = cy + int(h * 0.2)
    draw_dashed_ellipse(draw_img,
                        (cx - int(w * 0.22), shoulder_y + int(h * 0.1)),
                        (int(w * 0.18), int(h * 0.1)), 270, 360, gap_deg=4)
    draw_dashed_ellipse(draw_img,
                        (cx + int(w * 0.22), shoulder_y + int(h * 0.1)),
                        (int(w * 0.18), int(h * 0.1)), 180, 270, gap_deg=4)
    draw_dashed_line(draw_img,
                     (cx - int(w * 0.22), shoulder_y + int(h * 0.1)),
                     (cx - int(w * 0.22), h))
    draw_dashed_line(draw_img,
                     (cx + int(w * 0.22), shoulder_y + int(h * 0.1)),
                     (cx + int(w * 0.22), h))

    try:
        models     = get_ensemble_models()
        pil_img    = Image.fromarray(img_np).convert('RGB')
        img_tensor = img_transforms(pil_img).unsqueeze(0)
        meta       = torch.tensor([[gender_val]])

        bmi_raw_list = []
        with torch.no_grad():
            for model in models:
                bmi_norm, _ = model(img_tensor, meta)
                bmi_raw_list.append(bmi_norm.item())

        # 反歸一化（僅對 Female meta=0 校準）
        bmi_val = float(np.mean(bmi_raw_list)) * BMI_STD + BMI_MEAN
        bmi_val = float(np.clip(bmi_val, 10.0, 60.0))

        if bmi_val < 18.5:
            status_res = "🔵 體重過輕"
        elif bmi_val < 24.0:
            status_res = "🟢 健康體態"
        elif bmi_val < 27.0:
            status_res = "🟡 輕度過重"
        else:
            status_res = "🔴 肥胖體態"

    except Exception as e:
        bmi_val    = -1.0
        status_res = f"❌ 核心辨識異常: {str(e)}"
        st.error(f"🚨 模型運算發生錯誤：\n{traceback.format_exc()}")
    finally:
        gc.collect()

    return draw_img, bmi_val, status_res


# --- 7. CSS ---
st.markdown("""
<style>
    .stMarkdown h1 { color: #2E7D32; text-align: center; font-weight: bold; }
    .stMarkdown h3 { text-align: center; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- 8. UI ---
st.markdown("# 🧑‍⚕️ AI 臉部即時 BMI & 性別估算系統 by Jimmy Chen")
st.markdown("### 🎯 5-Fold 交叉驗證 Ensemble 統合（EfficientNet-B3 雙輸出頭版）")

# 性別選擇（meta 輸入）
gender_choice = st.radio(
    "👤 請先選擇性別（影響 BMI 計算精準度）：",
    ["♀️ 女性 (Female)", "♂️ 男性 (Male)"],
    horizontal=True
)
gender_val  = 0.0 if "女性" in gender_choice else 1.0
gender_label = "Female (女性)" if gender_val == 0.0 else "Male (男性)"

input_mode = st.radio(
    "👉 請選擇輸入方式：",
    ["📤 上傳本機照片", "📸 開啟鏡頭拍照"],
    horizontal=True
)

target_image = None
if input_mode == "📤 上傳本機照片":
    target_image = st.file_uploader(
        "選擇本機相簿中的正臉半身照片",
        type=["jpg", "jpeg", "png"],
        key="bmi_uploader_v3"
    )
else:
    target_image = st.camera_input("請將正臉與肩膀對齊畫面中央進行拍攝")

st.write("---")

# --- 9. 渲染 ---
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

    with st.spinner("🔍 5-Fold EfficientNet-B3 正在分析體態..."):
        res_draw, res_bmi, res_status = process_face_bmi(img_np, gender_val)

    with col_left:
        st.subheader("1. 臉部與對齊標記確認")
        if res_draw is not None:
            st.image(res_draw, use_container_width=True)

    with col_right:
        st.subheader("2. AI 綜合分析結果")

        st.subheader("📊 性別（使用者輸入）")
        st.info(f"**{gender_label}**")

        st.subheader("🩺 體態評估狀態")
        st.info(f"**{res_status}**")

        st.subheader("🎯 5-Fold 平均 BMI 值")
        st.metric(label="Ensemble Average BMI", value=f"{res_bmi:.2f}")

else:
    gc.collect()
    with col_left:
        st.info("💡 請先選擇性別，再上傳照片或開啟鏡頭拍照。")
    with col_right:
        st.text_input("📊 性別（使用者輸入）", value="等待輸入...", disabled=True, key="dis_gender_v3")
        st.text_input("🩺 體態評估狀態",       value="等待輸入...", disabled=True, key="dis_status_v3")
        st.subheader("🎯 5-Fold 平均 BMI 值")
        st.metric(label="Ensemble Average BMI", value="0.00")

st.markdown("---")
st.markdown(
    "<center>Developed by Jimmy Chen | 2026 Medical AI Track Edition</center>",
    unsafe_allow_html=True
)
