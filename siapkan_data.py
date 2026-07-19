import os
import pandas as pd
import random

# Konfigurasi folder dataset kopi lu
direktori_utama = 'train/'
kelas_kematangan = ['Dark', 'Green', 'Light', 'Medium']

# Variasi kalimat deskriptif agar AI belajar banyak pola
kamus_deskripsi = {
    'Dark': ["Biji kopi ini menunjukkan tingkat sangrai gelap dengan warna pekat dan permukaan yang memantulkan cahaya.",
             "Kondisi fisik kopi dark roast dengan warna cokelat tua kehitaman serta tekstur luar yang berminyak.",
             "Hasil sangrai maksimal menghasilkan biji berwarna gelap pekat dengan kandungan minyak yang keluar ke permukaan."],
    'Green': ["Citra ini menampilkan biji kopi mentah berwarna hijau pucat yang belum melewati proses penyangraian.",
              "Biji kopi dalam fase green bean dengan tekstur fisik yang sangat keras dan warna natural.",
              "Kondisi biji kopi yang belum disangrai sama sekali, memperlihatkan warna hijau khas biji mentah."],
    'Light': ["Biji kopi disangrai pada tingkat ringan, menghasilkan warna cokelat muda dengan permukaan yang sangat kering.",
              "Tingkat kematangan light roast terlihat dari warna cokelat pucat dan ketiadaan minyak di kulit biji.",
              "Proses sangrai singkat membentuk profil biji berwarna terang tanpa ada minyak yang terekspos."],
    'Medium': ["Gambar ini memperlihatkan kopi medium roast dengan warna cokelat karamel yang merata di seluruh permukaan.",
               "Tingkat sangrai menengah menghasilkan warna cokelat ideal dengan kondisi biji yang tetap kering tidak berminyak.",
               "Biji kopi berwarna cokelat sedang tanpa kemunculan minyak, menandakan profil kematangan medium roast."]
}

kumpulan_data = []

# Proses pencocokan gambar dengan kalimat
for kelas in kelas_kematangan:
    jalur_folder = os.path.join(direktori_utama, kelas)
    if os.path.exists(jalur_folder):
        daftar_gambar = os.listdir(jalur_folder)
        for nama_file in daftar_gambar:
            kalimat_terpilih = random.choice(kamus_deskripsi[kelas])
            kumpulan_data.append({
                'nama_gambar': f"{kelas}/{nama_file}",
                'teks_deskripsi': f"<start> {kalimat_terpilih} <end>"
            })

# Menyimpan ke dalam format CSV
dataframe_kopi = pd.DataFrame(kumpulan_data)
dataframe_kopi.to_csv('dataset_caption_kopi.csv', index=False)
print("Dokumen dataset_caption_kopi.csv berhasil diciptakan!")