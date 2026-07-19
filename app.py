import streamlit as st
import tensorflow as tf
import numpy as np
import random
from PIL import Image

# 1. Konfigurasi Halaman & Model
st.set_page_config(page_title="Kopi-AI Analytics", page_icon="☕")
st.title("☕ Kopi-AI Analytics")
st.write("Sistem Identifikasi Tingkat Kematangan Biji Kopi Arabika")

# Cache model agar tidak load ulang terus menerus
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_kopi_v1.h5')

model = load_my_model()
class_names = ['Dark', 'Green', 'Light', 'Medium']

# 2. Fungsi Captioning
def get_caption_ai(label):
    captions = {
        "Dark": [
            "Biji kopi berwarna hitam pekat, tekstur berminyak, rasa pahit kuat (Dark Roast).",
            "Hasil sangrai tingkat gelap dengan aroma smoky yang sangat dominan.",
            "Permukaan biji terlihat mengkilap karena minyak alami kopi telah keluar sempurna."
        ],
        "Green": [
            "Biji kopi mentah (Green Bean), tekstur sangat keras, belum melalui proses sangrai.",
            "Biji kopi dalam kondisi mentah dengan kadar air yang masih tinggi.",
            "Kopi belum disangrai, warna hijau pucat ini menandakan profil rasa yang masih segar."
        ],
        "Light": [
            "Biji kopi cokelat muda, aroma asam buah yang dominan, tanpa minyak (Light Roast).",
            "Kematangan rendah yang menjaga karakteristik asli (origin) dari biji kopi tersebut.",
            "Warna cokelat terang, permukaan kering, menghasilkan body kopi yang ringan."
        ],
        "Medium": [
            "Biji kopi cokelat sedang, keseimbangan antara rasa asam dan pahit (Medium Roast).",
            "Tingkat kematangan menengah, sangat populer karena profil rasa yang seimbang.",
            "Warna cokelat karamel merata dengan permukaan biji yang tetap kering."
        ]
    }
    return random.choice(captions.get(label, ["Menganalisis karakteristik biji..."]))

# 3. Antarmuka (UI)
uploaded_file = st.file_uploader("Upload citra biji kopi...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Buka gambar & paksa jadi RGB biar gak error shape
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Citra Input', use_column_width=True)
    
    if st.button('Mulai Analisis Sistem'):
        with st.spinner('Sedang menganalisis...'):
            # Preprocessing
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            predictions = model.predict(img_array)
            score = np.max(predictions) * 100
            label = class_names[np.argmax(predictions)]
            
            # Tampilkan Hasil
            st.success(f"Hasil Prediksi: **{label} Roast**")
            st.write(f"Tingkat Keyakinan: **{score:.2f}%**")
            
            # Narasi
            st.markdown("---")
            st.subheader("Narasi Karakteristik:")
            st.info(get_caption_ai(label))
