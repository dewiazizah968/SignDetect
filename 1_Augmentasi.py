import os
import random
from PIL import Image, ImageEnhance

input_data = 'Data Mentah Proyek PCD'
output_data = 'newdataset_mentah'
os.makedirs(output_data, exist_ok=True)

# Fungsi augmentasi
def augmentasi(image):
    augmented = []

    # Flip horizontal (mirror)
    flip_gambar = image.transpose(Image.FLIP_LEFT_RIGHT)
    augmented += [flip_gambar.copy() for _ in range(20)]

    # Rotasi (90, 180, 270)
    for angle in [90, 180, 270]:
        rotated = image.rotate(angle)
        augmented += [rotated.copy() for _ in range(20)]

    # Peningkatan cahaya
    for _ in range(20):
        factor = random.uniform(1.2, 1.8)
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(factor)
        augmented.append(brightened)

    return augmented

# Loop tiap kelas
for class_name in os.listdir(input_data):
    class_path = os.path.join(input_data, class_name)
    if not os.path.isdir(class_path):
        continue

    output_class_path = os.path.join(output_data, class_name)
    os.makedirs(output_class_path, exist_ok=True)

    list_gambar = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 3 sampel acak untuk augmentasi
    sample_images = random.sample(list_gambar, min(3, len(list_gambar)))

    for img_name in list_gambar:
        img_path = os.path.join(class_path, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            continue

        image.save(os.path.join(output_class_path, img_name))

        # Kalau sampel, di augmentasi
        if img_name in sample_images:
            aug_images = augmentasi(image)
            base_name = os.path.splitext(img_name)[0]
            for idx, aug_img in enumerate(aug_images):
                aug_img.save(os.path.join(output_class_path, f"{base_name}_aug{idx+1}.jpg"))

print("Augmentasi selesai. Dataset baru disimpan di 'newdataset_mentah'.")
