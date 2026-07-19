import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. KONFIGURASI MODEL & DATA
# ==========================================
# Load model yang sudah kamu buat (.h5)
model_path = 'model_kopi_v1.h5'
if not os.path.exists(model_path):
    print(f"❌ Error: File model '{model_path}' tidak ditemukan!")
    exit()

model = tf.keras.models.load_model(model_path)

# Daftar kelas (Harus urut abjad sesuai folder dataset)
class_names = ['Dark', 'Green', 'Light', 'Medium']

# Dictionary Deskripsi (Image Captioning Dasar)
descriptions = {
    "Dark": "Biji kopi berwarna hitam pekat, tekstur berminyak, rasa pahit kuat (Dark Roast).",
    "Green": "Biji kopi mentah (Green Bean), belum sangrai, tekstur sangat keras.",
    "Light": "Biji kopi cokelat muda, kering, tanpa minyak, aroma asam dominan (Light Roast).",
    "Medium": "Biji kopi cokelat sedang, keseimbangan rasa asam dan pahit (Medium Roast)."
}

# ==========================================
# 2. FUNGSI PREDIKSI & TAMPILKAN FOTO
# ==========================================
def prediksi_dan_tampilkan(img_path):
    # Cek apakah file gambar ada
    if not os.path.exists(img_path):
        print(f"❌ Error: File gambar '{img_path}' tidak ditemukan!")
        return

    # a. Load dan Preprocess Gambar untuk Model
    img_for_model = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img_for_model) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # b. Jalankan Prediksi
    predictions = model.predict(img_array)
    score = np.max(predictions) * 100
    label = class_names[np.argmax(predictions)]
    caption = descriptions.get(label, "Deskripsi tidak tersedia.")

    # c. Output ke Terminal (Sebagai backup log)
    print("\n" + "="*50)
    print(f"📷 HASIL ANALISIS FOTO: {label} ({score:.2f}%)")
    print(f"📝 DESKRIPSI (CAPTION): {caption}")
    print("="*50 + "\n")

    # d. MENAMPILKAN FOTO DENGAN HASIL (Permintaan Daffi)
    # Load gambar asli untuk ditampilkan
    img_to_show = tf.keras.utils.load_img(img_path)
    plt.figure(figsize=(8, 6)) # Atur ukuran jendela gambar
    
    # Tampilkan Gambar
    plt.imshow(img_to_show)
    
    # Tambahkan Judul (Hasil Klasifikasi)
    plt.title(f"Klasifikasi: {label} ({score:.2f}%)", fontsize=16, color='blue', fontweight='bold')
    
    # Sembunyikan Sumbu X dan Y (Biar bersih)
    plt.axis('off')
    
    # Tampilkan jendela gambar
    print("🔄 Membuka jendela gambar... (Tutup jendela untuk lanjut)")
    plt.show()

# ==========================================
# 3. JALANKAN TES
# ==========================================
# Masukkan path gambar kopi kamu di sini
# Contoh: kita ambil gambar dark (1).png dari folder test/Dark
path_gambar_tes = 'test/Dark/dark (1).png'

# Panggil fungsi
prediksi_dan_tampilkan(path_gambar_tes)