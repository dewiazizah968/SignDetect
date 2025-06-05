import os
import cv2
import numpy as np

input = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\newdataset_mentah'      
output = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\Data Biner'      # Folder hasil biner (otomatis)

os.makedirs(output, exist_ok=True)

for folder in os.listdir(input):
    input_folder = os.path.join(input, folder)
    output_folder = os.path.join(output, folder)

    if not os.path.isdir(input_folder):
        continue  

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        # Preprocessing
        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, cleaned)

print("Selesai preprocessing semua gambar ke dalam folder 'Data Biner'")
