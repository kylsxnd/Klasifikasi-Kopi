import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. Load model yang udah lu bikin
model = tf.keras.models.load_model('model_kopi_v1.h5')
class_names = ['Dark', 'Green', 'Light', 'Medium']

# 2. Siapin data test (PASTIKAN FOLDER INI ADA SESUAI LOKASI DATA LU)
# Ganti 'path_ke_folder_test_lu' dengan nama folder gambar kopi lu (misal: 'dataset/test')
test_dir = r'C:\Users\Daffi febrian\Documents\Penghitung Matang Kopi\test' 

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Jangan di-shuffle biar urutannya pas sama label
)

# 3. Suruh model nebak gambarnya
print("Model lagi nebak gambar, tunggu bentar...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# 4. Bikin Classification Report (Buat Sub-bab 3.6.1)
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=class_names))

# 5. Bikin Grafik Confusion Matrix (Buat Sub-bab 3.6.2)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Kopi AI')
plt.ylabel('Label Asli (True)')
plt.xlabel('Tebakan Model (Predicted)')
plt.show() # Ini bakal memunculkan jendela gambar, langsung lu Screenshot!