# Mengimpor algoritma Logistic Regression dari library Scikit-Learn (untuk klasifikasi)
from sklearn.linear_model import LogisticRegression

# Mengimpor library NumPy untuk manipulasi data numerik/array
import numpy as np

# --- TAHAP 1: PENYIAPAN DATA (DATA PREPARATION) ---
# Menyiapkan variabel independen (X) dalam bentuk 2D array (Matrix)
# Data ini mewakili 'Nilai Simulasi' mahasiswa
x = [[80], [40], [90], [50], [85]]

# Menyiapkan variabel dependen/target (y) 
# Label: 1 berarti Lulus, 0 berarti Gagal
y = [1, 0, 1, 0, 1]

# --- TAHAP 2: PEMBUATAN MODEL (MODELING) ---
# Menginisialisasi objek model Logistic Regression ke dalam variabel 'model'
model = LogisticRegression()

# Melatih model (Training) menggunakan data x dan y agar model bisa mempelajari pola hubungan
# Di sini model belajar: "Nilai berapa yang cenderung Lulus dan mana yang Gagal"
model.fit(x, y)

# --- TAHAP 3: INPUT INTERAKTIF (USER INTERFACE) ---
# Mengambil input nama dari user untuk personalisasi hasil
nama = input('Masukan Nama Mhs: ')

# Mengambil input nilai dan mengubah tipe datanya menjadi integer (angka bulat)
nilai_mhs = int(input('Masukan Nilai mhs: '))

# --- TAHAP 4: PREDIKSI & PROBABILITAS (INFERENCE) ---
# Melakukan prediksi kategori (0 atau 1) berdasarkan nilai input mahasiswa
# Data input dibungkus dalam kurung siku ganda [[ ]] karena model meminta format 2D
nilai = model.predict([[nilai_mhs]])

# Menghitung peluang (probabilitas) prediksi dalam rentang 0 sampai 1
# predict_proba menghasilkan [peluang_gagal, peluang_lulus]
# Kita mengambil index [0][1] untuk mendapatkan peluang Lulus
probabilitas = model.predict_proba([[nilai_mhs]])

# --- TAHAP 5: PENAMPILAN HASIL (OUTPUT) ---
# Menampilkan status kelulusan menggunakan teknik 'ternary operator' (Lulus jika nilai == 1)
print(f'\nMahasiswa Dengan Nama {nama} Dinyatakan {"Lulus" if nilai == 1 else "Gagal"}')

# Menampilkan persentase peluang lulus dengan format 2 angka di belakang desimal (:2f)
# Nilai probabilitas dikalikan 100 untuk mengubahnya menjadi persen
print(f'Peluang mahasiswa lulus adalah: {probabilitas[0][1] * 100:.2f}%')