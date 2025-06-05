import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\Data Mentah Proyek PCD\\A\\1.jpg'

img = cv2.imread(img_path)

img = cv2.resize(img, (128, 128))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding (Otsu)
_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Perbaikan morfologi
kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 3, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')

plt.subplot(1, 3, 3)
plt.imshow(cleaned, cmap='gray')
plt.title('Binary Mask')

plt.tight_layout()
plt.show()
