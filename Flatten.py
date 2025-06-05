import os
import cv2
import numpy as np
import csv

data_biner = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\Data Biner'
output_csv = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\data_flatten.csv'

# Buat header CSV
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['image_name'] + [f'pixel_{i}' for i in range(128 * 128)] + ['label']
    writer.writerow(header)

    # Iterasi gambar per folder
    for folder_name in os.listdir(data_biner):
        folder_path = os.path.join(data_biner, folder_name)

        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, file_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None or img.shape != (128, 128):
                    continue  # skip jika tidak terbaca atau ukurannya beda

                # Normalisasi dan flatten
                flattened = img.flatten() / 255.0
                image_name = f"{folder_name}_{file_name}"

                # Tulis ke CSV satu baris
                writer.writerow([image_name] + flattened.tolist() + [folder_name])

print("CSV berhasil dibuat")
