import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

data_biner = 'Data Biner'
save_dir = 'saved_model_SVM'
os.makedirs(save_dir, exist_ok=True)

data = []
labels = []

for folder in os.listdir(data_biner):
    folder_path = os.path.join(data_biner, folder)
    if not os.path.isdir(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (128, 128))
            flat = img.flatten() / 255.0
            data.append(flat)
            labels.append(folder)

X = np.array(data, dtype=np.float32)
y_raw = np.array(labels)

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# PCA buat reduksi dimensi
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# model SVM
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train_pca, y_train)

# prediksi
y_pred = svm.predict(X_test_pca)

print("\n=== HASIL EVALUASI SVM ===")
print("Akurasi:", accuracy_score(y_test, y_pred))
print("Laporan Klasifikasi:\n", classification_report(y_test, y_pred, target_names=le.classes_))

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

joblib.dump(svm, os.path.join(save_dir, 'svm_model.joblib'))
joblib.dump(pca, os.path.join(save_dir, 'pca_model.joblib'))
joblib.dump(le, os.path.join(save_dir, 'label_encoder.joblib'))

print("\nModel SVM, PCA, dan LabelEncoder berhasil disimpan ke folder 'saved_model_SVM'.")