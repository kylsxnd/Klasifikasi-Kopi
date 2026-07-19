import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 1. KONFIGURASI PATH DAN PARAMETER
folder_gambar = 'train/' 
file_csv = 'dataset_caption_kopi.csv'
panjang_maksimal = 30
ukuran_kosakata = 5000

# 2. PENANGANAN GPU (Meskipun Windows Native seringkali lari ke CPU)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("🚀 Konfigurasi GPU berhasil dilakukan.")
    except RuntimeError as e:
        print(e)

# 3. MEMUAT DATASET
print("Sedang memproses dataset...")
df = pd.read_csv(file_csv)
kumpulan_gambar = df['nama_gambar'].tolist()
kumpulan_caption = df['teks_deskripsi'].tolist()

# 4. TOKENISASI TEKS
tokenizer = Tokenizer(num_words=ukuran_kosakata, oov_token="<unk>")
tokenizer.fit_on_texts(kumpulan_caption)
ukuran_kosakata_asli = len(tokenizer.word_index) + 1

# 5. DATA GENERATOR (VERSI PERBAIKAN TUPLE)
def data_generator(daftar_gambar, daftar_caption, folder, tokenizer, panjang_max, batch_size=32):
    X1, X2, y = list(), list(), list()
    n = 0
    while True:
        for i in range(len(daftar_caption)):
            n += 1
            path_gambar = os.path.join(folder, daftar_gambar[i])
            try:
                img = load_img(path_gambar, target_size=(224, 224))
                img_array = img_to_array(img) / 255.0
            except:
                continue
            
            seq = tokenizer.texts_to_sequences([daftar_caption[i]])[0]
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                in_seq = pad_sequences([in_seq], maxlen=panjang_max)[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=ukuran_kosakata_asli)[0]
                
                X1.append(img_array)
                X2.append(in_seq)
                y.append(out_seq)
            
            if n == batch_size:
                # PERBAIKAN: Menggunakan format tuple ((x1, x2), y)
                yield ((np.array(X1), np.array(X2)), np.array(y))
                X1, X2, y = list(), list(), list()
                n = 0

# 6. MEMBANGUN ARSITEKTUR MODEL
print("Membangun arsitektur hibrida CNN-LSTM...")
input_citra = Input(shape=(224, 224, 3))
model_penglihatan = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model_penglihatan.trainable = False

fitur_citra = model_penglihatan(input_citra)
fitur_citra = Dense(256, activation='relu')(fitur_citra)
fitur_citra = Dropout(0.5)(fitur_citra)

input_teks = Input(shape=(panjang_maksimal,))
penyematan_kata = Embedding(ukuran_kosakata_asli, 256, mask_zero=True)(input_teks)
penyematan_kata = Dropout(0.5)(penyematan_kata)
proses_lstm = LSTM(256)(penyematan_kata)

penggabungan = add([fitur_citra, proses_lstm])
lapisan_tersembunyi = Dense(256, activation='relu')(penggabungan)
output_kata = Dense(ukuran_kosakata_asli, activation='softmax')(lapisan_tersembunyi)

model = Model(inputs=[input_citra, input_teks], outputs=output_kata)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# 7. PROSES TRAINING
print("Memulai proses pelatihan model...")
epochs = 30
steps = len(kumpulan_caption) // 32
generator = data_generator(kumpulan_gambar, kumpulan_caption, folder_gambar, tokenizer, panjang_maksimal, 32)

model.fit(generator, epochs=epochs, steps_per_epoch=steps, verbose=1)

# 8. PENYIMPANAN MODEL
model.save('model_kopi_captioning.h5')
# Menyimpan kamus kata agar sinkron dengan aplikasi web
import pickle
with open('tokenizer_kopi.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Proses selesai. Model dan tokenizer telah tersimpan.")