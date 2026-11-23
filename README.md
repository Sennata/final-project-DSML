# ClothValuate - Prediksi Harga Baju Thrift (Final Project DSML)

Aplikasi web berbasis HTML, Tailwind CSS, dan Flask untuk memprediksi harga pakaian bekas / thrift berdasarkan model Machine Learning Random Forest (`random_forest_model.joblib`).

---

## 📌 Fitur Aplikasi
- **Antarmuka Modern & Responsif**: Menggunakan Tailwind CSS, kartu glassmorphism, dan visual selector (radio card & pill badge).
- **Koneksi Langsung ke Model Scikit-Learn**: Membaca pipeline preprocessing (`ColumnTransformer`, `OneHotEncoder`, `SimpleImputer`) dan `RandomForestRegressor`.
- **Hasil Prediksi Instan**:
  - Taksiran harga dalam Rupiah (misal: `Rp 50.455`).
  - Rentang toleransi harga berdasarkan MAE model (`± Rp 13.024`).
  - Ringkasan fitur yang dipilih secara real-time.
- **Preset Uji Cepat**: Tombol untuk memuat contoh pakaian (Kaos Giordano Grade A, Polo The Executive, Pakaian Thrift Non-Merk).

---

## 📊 Atribut Input Model (Berdasarkan `Final_Project_DSML.ipynb`)
Model menerima 8 fitur kategorikal:
1. `Ukuran`: S, M, L, XL, All Size.
2. `Kelayakan (Grade)`: Grade A (Like new), Grade B (Baik), Grade C (Ada cacat/minus).
3. `Kondisi`: `-` (Mulus/Normal), `Ada noda kecil`, `Warna sedikit pudar`, `Kancing beda 1`, dll.
4. `Tipe Kain`: Katun, Polyester, Rayon, Flanel, Baby Terry, Linen, Knit, Scuba, dll.
5. `Tipe Baju`: Kaos (T-Shirt), Kerah (Polo / Kemeja).
6. `Merk (Merk/Non-Merk`: Non-Merk, Merk (Giordano), Merk (The Excecutive), Merk (Uniqlo), dll.
7. `Polosan/Gambar (Y/N)`: Polos, Gambar, Motif Garis-garis, Kotak-kotak, Batik, dll.
8. `Gender (M/F/U)`: M (Pria), F (Wanita), U (Unisex).

---

## 🚀 Cara Menjalankan Aplikasi

### Opsi 1: Klik Dua Kali (Windows)
Cukup jalankan file `run.bat` dengan cara double click.

### Opsi 2: Menggunakan Terminal
```bash
python app.py
```
Lalu buka browser Anda di:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 📈 Metrik Evaluasi Model (Dari Notebook)
- **Model Terbaik**: Random Forest Regressor (100 Trees)
- **R² Score**: `0.5652`
- **Mean Absolute Error (MAE)**: `Rp 13.024,72`
- **Root Mean Squared Error (RMSE)**: `Rp 19.038,40`
