import streamlit as st
from PIL import Image
import cv2
import numpy as np
import joblib
import os

knn_path = "saved_model_KNN"
svm_path = "saved_model_SVM"

# Load model KNN
pca_knn = joblib.load(os.path.join(knn_path, "pca_model.joblib"))
knn = joblib.load(os.path.join(knn_path, "knn_model.joblib"))
le_knn = joblib.load(os.path.join(knn_path, "label_encoder.joblib"))

# Load model SVM
pca_svm = joblib.load(os.path.join(svm_path, "pca_model.joblib"))
svm = joblib.load(os.path.join(svm_path, "svm_model.joblib"))
le_svm = joblib.load(os.path.join(svm_path, "label_encoder.joblib"))

# Resize dan crop tengah
def resize_crop_center(img, target_size=(128, 128)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped

# Preprocessing
def process_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = resize_crop_center(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    flatten = cleaned.flatten().astype(np.float32) / 255.0
    return img, cleaned, flatten

# Prediksi dengan dua model
def predict_all(flatten):
    # PCA & prediksi KNN
    flat_knn = pca_knn.transform([flatten])
    pred_knn = knn.predict(flat_knn)
    label_knn = le_knn.inverse_transform(pred_knn)[0]

    # PCA & prediksi SVM
    flat_svm = pca_svm.transform([flatten])
    pred_svm = svm.predict(flat_svm)
    label_svm = le_svm.inverse_transform(pred_svm)[0]

    return label_knn, label_svm

# === Streamlit GUI ===
st.set_page_config(page_title="Prediksi Gestur Tangan", layout="centered")
st.markdown("<h1 style='text-align: center;'>🖐️ Prediksi Gestur Tangan</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: gray;'>(KNN & SVM)</h3>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload gambar (.jpg, .png, .jpeg)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_img, binary_img, flatten = process_image(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar Asli")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), channels="RGB", width=250)
    with col2:
        st.subheader("Gambar Biner")
        st.image(binary_img, channels="GRAY", width=250)

    col_pred = st.columns([1, 2, 1])[1]  
    with col_pred:
        predict_clicked = st.button("🔍 Prediksi", use_container_width=True)

    if predict_clicked:
        label_knn, label_svm = predict_all(flatten)

        col3, col4 = st.columns(2)
        with col3:
            st.markdown(f"<h3 style='text-align:center;'>🎯 KNN</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align:center; font-size:32px; color:green;'><strong>{label_knn}</strong></p>",
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(f"<h3 style='text-align:center;'>🎯 SVM</h3>", unsafe_allow_html=True)
            st.markdown(
                f"<p style='text-align:center; font-size:32px; color:blue;'><strong>{label_svm}</strong></p>",
                unsafe_allow_html=True
            )
