import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data_path = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\Data Biner'
image_data = []
labels = []

for folder_name in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder_name)
    if not os.path.isdir(folder_path):
        continue

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None or img.shape != (128, 128):
                continue

            flattened = img.flatten() / 255.0
            image_data.append(flattened)
            labels.append(folder_name)

# Konversi ke array
X = np.array(image_data, dtype=np.float32)
y = np.array(labels)

# Encode label ke angka
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_pca, y_train)
y_pred = knn.predict(X_test_pca)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Simpan model dan komponen
save_dir = 'D:\\PYTHON PROJECT\\Data Science UNESA\\Semester 4\\PCD\\saved_model_KNN'
os.makedirs(save_dir, exist_ok=True)
joblib.dump(pca, os.path.join(save_dir, 'pca_model.joblib'))
joblib.dump(le, os.path.join(save_dir, 'label_encoder.joblib'))
joblib.dump(knn, os.path.join(save_dir, 'knn_model.joblib'))
print("Model dan komponen disimpan.")
