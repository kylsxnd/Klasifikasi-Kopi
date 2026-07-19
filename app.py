from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
import random

app = Flask(__name__)

# Load model AI kamu
model = tf.keras.models.load_model('model_kopi_v1.h5')
class_names = ['Dark', 'Green', 'Light', 'Medium']

# Fungsi Image Captioning Dinamis
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            if not os.path.exists('static'): os.makedirs('static')
            path = os.path.join('static', file.filename)
            file.save(path)
            
            # Preprocessing Gambar
            img = tf.keras.utils.load_img(path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi AI
            predictions = model.predict(img_array)
            score = np.max(predictions) * 100
            label = class_names[np.argmax(predictions)]
            
            # Ambil Caption & Format Skor
            caption_final = get_caption_ai(label)
            score_final = "{:.2f}".format(score)
            
            return render_template('index.html', 
                                   label=label, 
                                   caption=caption_final, 
                                   img_path=path.replace("\\", "/"), 
                                   accuracy=score_final)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)