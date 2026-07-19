import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model

def bangun_arsitektur_captioning(ukuran_kosakata, panjang_maksimal_kalimat):
    # 1. Fase Ekstraksi Visual (MATA)
    input_citra = Input(shape=(224, 224, 3), name="jalur_input_gambar")
    model_penglihatan = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    model_penglihatan.trainable = False
    
    fitur_citra = model_penglihatan(input_citra)
    fitur_citra_diproses = Dense(256, activation='relu', name="ekstraksi_fitur_padat")(fitur_citra)
    fitur_citra_diproses = Dropout(0.5)(fitur_citra_diproses)

    # 2. Fase Pemrosesan Bahasa (MULUT)
    input_teks = Input(shape=(panjang_maksimal_kalimat,), name="jalur_input_teks")
    penyematan_kata = Embedding(ukuran_kosakata, 256, mask_zero=True, name="penyematan_bahasa")(input_teks)
    penyematan_kata = Dropout(0.5)(penyematan_kata)
    proses_lstm = LSTM(256, name="memori_bahasa_lstm")(penyematan_kata)

    # 3. Fase Penggabungan Hibrida (ENCODER-DECODER)
    penggabungan_sistem = add([fitur_citra_diproses, proses_lstm])
    lapisan_tersembunyi = Dense(256, activation='relu')(penggabungan_sistem)
    output_kalimat = Dense(ukuran_kosakata, activation='softmax', name="prediksi_kata")(lapisan_tersembunyi)

    # Menyegel arsitektur
    model_hibrida = Model(inputs=[input_citra, input_teks], outputs=output_kalimat)
    model_hibrida.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model_hibrida

# Inisialisasi kerangka model
model_ai_kopi = bangun_arsitektur_captioning(ukuran_kosakata=5000, panjang_maksimal_kalimat=30)
model_ai_kopi.summary()