import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kopi-AI System", page_icon="☕", layout="centered")

# --- CUSTOM PROFESSIONAL CSS ---
st.markdown("""
    <style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif;
        background-color: #1B1210; /* Espresso Dark */
    }

    /* Header Styling */
    header, [data-testid="stHeader"] {
        background-color: rgba(27, 18, 16, 0.9) !important;
    }

    /* Sidebar Professional Look */
    [data-testid="stSidebar"] {
        background-color: #2D1B18 !important;
        border-right: 1px solid #3E2723;
    }

    /* Card Layout untuk Hasil */
    .report-card {
        background-color: #2D1B18;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #4E342E;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-top: 20px;
    }
    
    /* Card Layout untuk Error/Bukan Kopi */
    .error-card {
        background-color: #2D1B18;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #D32F2F;
        border-left: 5px solid #D32F2F;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-top: 20px;
    }

    /* Deskripsi Styling (Yang disuruh dosen diperbesar) */
    .caption-text {
        font-size: 24px !important; 
        line-height: 1.6;
        font-weight: 300;
        color: #D7CCC8 !important; /* Latte White */
        background: rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #6D4C41;
        margin-top: 15px;
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6D4C41 0%, #3E2723 100%);
        color: white !important;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.4);
        border: 1px solid #D7CCC8;
    }

    /* Text & Labels */
    h1, h2, h3 { color: #D7CCC8 !important; }
    p, span, label { color: #BCAAA4 !important; }
    
    /* Progress Bar Color */
    .stProgress > div > div > div > div {
        background-color: #6D4C41;
    }

    /* Hilangkan Toolbar */
    [data-testid="stToolbar"] { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('model_kopi_v1.h5')

model = load_my_model()
class_names = ['Dark', 'Green', 'Light', 'Medium']

# --- RULE-BASED CAPTIONING ---
def get_caption(label):
    captions = {
        "Dark": [
            "Biji kopi berwarna hitam pekat, tekstur berminyak, rasa pahit kuat (Dark Roast).",
            "Hasil sangrai tingkat gelap dengan aroma smoky yang sangat dominan.",
            "Permukaan biji mengkilap karena minyak alami telah keluar sempurna."
        ],
        "Green": [
            "Biji kopi mentah (Green Bean), belum melalui proses sangrai.",
            "Biji kopi dalam kondisi mentah dengan kadar air yang masih tinggi.",
            "Warna hijau pucat ini menandakan profil rasa yang masih segar."
        ],
        "Light": [
            "Biji kopi cokelat muda, aroma asam buah yang dominan (Light Roast).",
            "Kematangan rendah yang menjaga karakteristik asli (origin) biji kopi.",
            "Permukaan kering tanpa minyak dengan body kopi yang ringan."
        ],
        "Medium": [
            "Biji kopi cokelat sedang, keseimbangan rasa asam dan pahit (Medium Roast).",
            "Tingkat menengah yang populer karena profil rasa yang sangat seimbang.",
            "Warna cokelat karamel merata dengan permukaan biji yang kering."
        ]
    }
    return random.choice(captions[label])

# --- UI MAIN ---
st.markdown("<h1 style='text-align: center;'>☕ KOPI-AI ANALYTICS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sistem Identifikasi Tingkat Penyangraian & Image Captioning Otomatis</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    try:
        st.image("logo_gundar.png", width=120)
    except:
        st.subheader("Universitas Gunadarma")
    
    st.markdown("### Detail Project")
    st.info("**Penulisan Ilmiah**\n\nTeknik Informatika")
    st.markdown("---")
    st.write("**Model:** CNN MobileNetV2")
    st.write("**Method:** Rule-based Mapping")

# Upload Area
uploaded_file = st.file_uploader("Upload citra biji kopi...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        # Gambar dikecilkan sesuai saran dosen
        st.image(image, caption='Citra Input', width=280)
    
    if st.button('Mulai Analisis Sistem'):
        with st.spinner('Mengekstraksi fitur visual...'):
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = model.predict(img_array)
            score = np.max(predictions) * 100
            label = class_names[np.argmax(predictions)]
            
            # --- LOGIKA THRESHOLDING (PENANGANAN BUKAN KOPI ATAU GAMBAR BURUK) ---
            THRESHOLD = 75.0 # Ambang batas 75%
            
            if score < THRESHOLD:
                # --- HASIL JIKA BUKAN KOPI (DI BAWAH THRESHOLD) ---
                st.markdown(f"""
                    <div class="error-card">
                        <h3 style='margin:0; color:#EF5350;'>⚠️ Objek Tidak Dikenali!</h3>
                        <p style='margin-bottom:10px;'>Tingkat Keyakinan: {score:.2f}%</p>
                        <p style='color:#BCAAA4; font-size:14px;'>Sistem mendeteksi bahwa gambar yang diunggah kemungkinan besar <b>BUKAN biji kopi</b>, atau gambar terlalu buram/tidak menggunakan latar belakang putih. Silakan unggah gambar yang sesuai.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # --- HASIL DALAM CARD (JIKA KOPI VALID) ---
                st.markdown(f"""
                    <div class="report-card">
                        <h3 style='margin:0;'>Hasil Analisis: <span style='color:#D7CCC8;'>{label} Roast</span></h3>
                        <p style='margin-bottom:10px;'>Tingkat Keyakinan: {score:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.progress(int(score))
                
                # --- DESKRIPSI DIBAWAH ---
                st.markdown("<br><b>Narasi Karakteristik Biji:</b>", unsafe_allow_html=True)
                st.markdown(f'<div class="caption-text"><i>"{get_caption(label)}"</i></div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 0.8rem; opacity: 0.6;'>© 2026 Daffi Febrian - Universitas Gunadarma</p>", unsafe_allow_html=True)