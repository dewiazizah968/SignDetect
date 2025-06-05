import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib

# Load model PCA, KNN, dan Label Encoder
pca = joblib.load('D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\saved_model_KNN\\pca_model.joblib')
knn = joblib.load('D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\saved_model_KNN\\knn_model.joblib')
le = joblib.load('D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\saved_model_KNN\\label_encoder.joblib')

def resize_crop_center(img, target_size=(128, 128)):
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Crop bagian tengah
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    return cropped

# Pre-processing
def process_and_predict(image_path):
    img = cv2.imread(image_path)

    img = resize_crop_center(img, (128, 128))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    flatten = cleaned.flatten().astype(np.float32) / 255.0
    flatten_pca = pca.transform([flatten])
    pred = knn.predict(flatten_pca)
    label = le.inverse_transform(pred)[0]

    return img, cleaned, label

# Upload dan prediksi
def upload_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        original_img, processed_img, prediction = process_and_predict(file_path)

        # Gambar asli
        ori = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ori = ori.resize((200, 200))
        ori_tk = ImageTk.PhotoImage(ori)
        original_label.config(image=ori_tk)
        original_label.image = ori_tk

        # Gambar hasil biner
        proc = Image.fromarray(processed_img)
        proc = proc.resize((200, 200))
        proc_tk = ImageTk.PhotoImage(proc)
        binary_label.config(image=proc_tk)
        binary_label.image = proc_tk

        # Hasil prediksi
        pred_label.config(text=f"Prediksi: {prediction}")

    except Exception as e:
        messagebox.showerror("Error", f"Gagal memproses gambar: {e}")

# === GUI Layout ===
root = tk.Tk()
root.title("Prediksi Gestur Tangan")

# Frame utama
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack()

# Frame kiri 
left_frame = tk.Frame(main_frame)
left_frame.grid(row=0, column=0, padx=10)

original_label = tk.Label(left_frame, text="Gambar yang diupload", bg="lightgray")
original_label.pack()

upload_btn = tk.Button(left_frame, text="Upload Gambar", command=upload_and_predict)
upload_btn.pack(pady=10)

# Frame kanan 
right_frame = tk.Frame(main_frame)
right_frame.grid(row=0, column=1, padx=10)

binary_label = tk.Label(right_frame, text="Gambar hasil Biner", bg="lightgray")
binary_label.pack()

pred_label = tk.Label(right_frame, text="Hasil prediksi", font=('Arial', 12))
pred_label.pack(pady=10)

root.mainloop()
