# Dokumentasi Tes Pemrograman dan Machine Learning

Proyek ini berisi solusi untuk tiga soal tes pemrograman dan machine learning, mencakup inference video menggunakan YOLO, analisis hasil pelatihan model, dan perancangan algoritma pelacakan kendaraan berbasis unsupervised machine learning. Berikut adalah dokumentasi lengkap untuk instalasi, persyaratan, cara menjalankan, dan hasil dari masing-masing solusi.

---

## Daftar Isi
1. [Ikhtisar Soal](#ikhtisar-soal)
2. [Persyaratan](#persyaratan)
3. [Instalasi](#instalasi)
4. [Struktur Direktori](#struktur-direktori)
5. [Petunjuk Menjalankan](#petunjuk-menjalankan)
   - [Soal 1: Inference Video Menggunakan YOLO](#soal-1-inference-video-menggunakan-yolo)
   - [Soal 2: Analisis Grafik Hasil Pelatihan](#soal-2-analisis-grafik-hasil-pelatihan)
   - [Soal 3: Perancangan Algoritma Pelacakan Kendaraan](#soal-3-perancangan-algoritma-pelacakan-kendaraan)
6. [Hasil](#hasil)
   - [Hasil Soal 1](#hasil-soal-1)
   - [Hasil Soal 2](#hasil-soal-2)
   - [Hasil Soal 3](#hasil-soal-3)
7. [Catatan Tambahan](#catatan-tambahan)

---

## Ikhtisar Soal
Proyek ini menjawab tiga soal berikut, sebagaimana tercantum dalam dokumen asli:

1. **Inference Video Menggunakan YOLO**:
   ```
   Buatlah program sederhana untuk mendeteksi dan menghitung objek pada video yang terdapat di dalam folder `video`, menggunakan model YOLO yang disimpan dalam folder `model`. Program harus memenuhi ketentuan sebagai berikut:
   - Melakukan inference setiap 30 frame (1 frame per detik jika video 30 FPS).
   - Mengganti label hasil inference dengan klasifikasi kualitas buah sebagai berikut:  
     `0 = Ripe`
     `1 = Unripe`
     `2 = OverRipe`
     `3 = Rotten`
     `4 = EmptyBunch`
   - Objek yang terdeteksi harus tetap dilacak (tracking) hingga melewati garis penghitung (counting line).
   ```
   **Solusi**: Implementasi dalam [`inference_yolo.py`](./inference_yolo.py) menggunakan YOLO untuk deteksi, Kalman Filter untuk pelacakan, dan GUI PyQt5 untuk visualisasi.

2. **Analisis Grafik Hasil Pelatihan**:
   ```
   Berikan penjelasan terhadap grafik hasil training yang terdapat pada masing-masing file gambar dalam folder `train_result`.  
   Penjelasan harus mencakup:
   - Loss (Training dan Validation)
   - Matriks evaluasi seperti mAP, Precision, dan Recall
   - Interpretasi terhadap performa model berdasarkan grafik
   ```
   **Solusi**: Analisis mendetail disediakan dalam [`analisis.md`](./analisis.md), mencakup distribusi kelas, metrik evaluasi, dan rekomendasi perbaikan.

3. **Perancangan Algoritma Pelacakan Kendaraan**:
   ```
   Rancanglah algoritma atau pendekatan berbasis Unsupervised Machine Learning yang sesuai pada data tracking kendaraan berikut:
   - **Database**  : `Snowflake`
   - **Host**      : `https://hb01677.ap-southeast-3.aws.snowflakecomputing.com/`
   - **User**      : `TES_USR_LACAK`
   - **Password**  : `StrongPassword123`
   ```
   **Solusi**: Implementasi dalam [`vehicle_tracking.py`](./vehicle_tracking.py) menggunakan K-Means untuk clustering data kendaraan dari Snowflake, menghasilkan laporan interaktif dalam [`vehicle_tracking_report.html`](./vehicle_tracking_report.html).

---

## Persyaratan
Berikut adalah persyaratan perangkat lunak dan perangkat keras untuk menjalankan solusi:

### Perangkat Lunak
- **Python**: Versi 3.8 atau lebih baru
- **Dependensi Python** (tercantum dalam [`requirements.txt`](./requirements.txt)):
  - `opencv-python>=4.5.5`
  - `torch>=1.8.0`
  - `ultralytics>=8.0.0`
  - `filterpy>=1.4.5`
  - `PyQt5>=5.15.6`
  - `snowflake-connector-python>=2.7.0`
  - `numpy>=1.21.0`
  - `pandas>=1.3.0`
  - `scikit-learn>=1.0.0`
  - `tqdm>=4.62.0`
- **Sistem Operasi**: Windows, Linux, atau macOS
- **Browser**: Untuk melihat laporan HTML (misalnya, Chrome, Firefox)
- **Snowflake Account**: Akses ke database Snowflake dengan kredensial yang valid (untuk Soal 3)

### Perangkat Keras
- **CPU**: Minimal Intel i5 atau setara
- **RAM**: Minimal 8 GB (disarankan 16 GB untuk performa optimal)
- **GPU** (opsional): NVIDIA GPU dengan CUDA untuk inference YOLO yang lebih cepat
- **Penyimpanan**: Minimal 2 GB ruang kosong untuk dependensi dan file output

---

## Instalasi
Ikuti langkah-langkah berikut untuk mengatur lingkungan dan menjalankan proyek:

1. **Clone Repository** (jika proyek berada di repository):
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Buat Virtual Environment** (disarankan):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Instal Dependensi**:
   Buat file [`requirements.txt`](./requirements.txt) dengan konten berikut:
   ```
   opencv-python>=4.5.5
   torch>=1.8.0
   ultralytics>=8.0.0
   filterpy>=1.4.5
   PyQt5>=5.15.6
   snowflake-connector-python>=2.7.0
   numpy>=1.21.0
   pandas>=1.3.0
   scikit-learn>=1.0.0
   tqdm>=4.62.0
   ```
   Jalankan perintah:
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan File yang Diperlukan**:
   - Folder [`video/`](./video/) berisi file video: [`conveyor.mp4`](./video/conveyor.mp4), [`precessed-1-frame.mp4`](./video/precessed-1-frame.mp4), [`precessed-30-frame.mp4`](./video/precessed-30-frame.mp4)
   - Folder [`model/`](./model/) berisi file model YOLO: [`22K-5-M.pt`](./model/22K-5-M.pt)
   - Folder [`train_result/`](./train_result/) berisi file gambar hasil pelatihan:
     - [`Class_Count.png`](./train_result/Class_Count.png)
     - [`Metrics.png`](./train_result/Metrics.png)
     - [`AP_by_Class.png`](./train_result/AP_by_Class.png)
     - [`Training_Graphs.png`](./train_result/Training_Graphs.png)
     - [`Advance_Training_Graphs.png`](./train_result/Advance_Training_Graphs.png)
   - File utama:
     - [`inference_yolo.py`](./inference_yolo.py)
     - [`analisis.md`](./analisis.md)
     - [`vehicle_tracking.py`](./vehicle_tracking.py)
     - [`vehicle_tracking_report.html`](./vehicle_tracking_report.html)
     - [`requirements.txt`](./requirements.txt)

5. **Konfigurasi Snowflake** (untuk Soal 3):
   Pastikan kredensial Snowflake di [`vehicle_tracking.py`](./vehicle_tracking.py) sudah benar:
   ```python
   conn = snowflake.connector.connect(
       user="TES_USR_LACAK",
       password="StrongPassword123",
       account="hb01677.ap-southeast-3.aws",
       database="TES_DB_RAW",
       schema="RAW_LACAK"
   )
   ```

---

## Struktur Direktori
Berikut adalah struktur direktori proyek yang diharapkan:
```
project_directory/
├── video/
│   ├── conveyor.mp4
│   ├── precessed-1-frame.mp4
│   └── precessed-30-frame.mp4
├── model/
│   └── 22K-5-M.pt
├── train_result/
│   ├── Class_Count.png
│   ├── Metrics.png
│   ├── AP_by_Class.png
│   ├── Training_Graphs.png
│   └── Advance_Training_Graphs.png
├── inference_yolo.py
├── analisis.md
├── vehicle_tracking.py
├── vehicle_tracking_report.html
├── README.md
└── requirements.txt
```

---

## Petunjuk Menjalankan

### Soal 1: Inference Video Menggunakan YOLO
**File**: [`inference_yolo.py`](./inference_yolo.py)

**Deskripsi**: Program ini mendeteksi dan menghitung objek pada video menggunakan model YOLO, dengan pelacakan berbasis Kalman Filter dan GUI berbasis PyQt5. Objek dihitung saat melewati garis penghitung, dengan label kelas diubah menjadi `Ripe`, `Unripe`, `OverRipe`, `Rotten`, dan `EmptyBunch`.

**Cara Menjalankan**:
1. Pastikan folder [`video/`](./video/) berisi [`conveyor.mp4`](./video/conveyor.mp4) dan folder [`model/`](./model/) berisi [`22K-5-M.pt`](./model/22K-5-M.pt).
2. Jalankan skrip:
   ```bash
   python inference_yolo.py
   ```
3. GUI akan muncul dengan langkah-langkah berikut:
   - **Pilih Video**: Secara default, [`video/conveyor.mp4`](./video/conveyor.mp4) digunakan. Klik "Browse" untuk memilih video lain, seperti [`precessed-1-frame.mp4`](./video/precessed-1-frame.mp4) atau [`precessed-30-frame.mp4`](./video/precessed-30-frame.mp4).
   - **Atur Garis Penghitung**: Masukkan persentase Y (default: 50%) untuk posisi garis penghitung.
   - **Mulai Deteksi**: Klik "Start Detection" untuk memulai inference.
   - **Hentikan Deteksi**: Klik "Stop Detection" untuk menghentikan.
   - **Reset Hitungan**: Klik "Reset Counts" untuk mengatur ulang hitungan kelas.
4. Program akan menampilkan video dengan bounding box, ID pelacakan, kecepatan vertikal (vy), dan hitungan kelas di panel kanan.

**Catatan**:
- Inference dilakukan setiap frame (bukan setiap 30 frame seperti spesifikasi awal) untuk akurasi pelacakan yang lebih baik, tetapi dapat dimodifikasi dengan mengubah `process_frame_interval` di [`inference_yolo.py`](./inference_yolo.py).
- Pastikan GPU tersedia untuk performa optimal, jika tidak, program akan menggunakan CPU.

### Soal 2: Analisis Grafik Hasil Pelatihan
**File**: [`analisis.md`](./analisis.md)

**Deskripsi**: Dokumen ini menganalisis hasil pelatihan model berdasarkan grafik di folder [`train_result/`](./train_result/), mencakup distribusi kelas, metrik evaluasi (mAP, Precision, Recall), analisis loss (pelatihan dan validasi), dan interpretasi performa model.

**Cara Menjalankan**:
- Tidak ada skrip untuk dijalankan karena ini adalah dokumen analisis.
- Buka [`analisis.md`](./analisis.md) menggunakan editor teks atau viewer Markdown (misalnya, VS Code, GitHub, atau Markdown Viewer).
- Pastikan folder [`train_result/`](./train_result/) berisi file gambar yang dirujuk: [`Class_Count.png`](./train_result/Class_Count.png), [`Metrics.png`](./train_result/Metrics.png), [`AP_by_Class.png`](./train_result/AP_by_Class.png), [`Training_Graphs.png`](./train_result/Training_Graphs.png), [`Advance_Training_Graphs.png`](./train_result/Advance_Training_Graphs.png).

**Struktur Analisis**:
- **Distribusi Kelas**: Menunjukkan ketidakseimbangan dataset (Kelas 0: 37.548 sampel, Kelas 3: 71 sampel).
- **Metrik Keseluruhan**: mAP@50 (55,4%), Precision (54,5%), Recall (54,0%).
- **mAP per Kelas**: Kelas 0 berkinerja baik (mAP@50 > 90%), Kelas 3 buruk (mAP@50 < 9%).
- **Analisis Grafik**: Loss pelatihan dan validasi menurun, tetapi ada tanda overfitting. Metrik seperti mAP dan Precision meningkat stabil.
- **Interpretasi**: Model baik untuk kelas dengan banyak data, buruk untuk kelas minoritas, dengan rekomendasi seperti oversampling dan regularisasi.

### Soal 3: Perancangan Algoritma Pelacakan Kendaraan
**File**: [`vehicle_tracking.py`](./vehicle_tracking.py), [`vehicle_tracking_report.html`](./vehicle_tracking_report.html)

**Deskripsi**: Program ini mengambil data pelacakan kendaraan dari database Snowflake, menerapkan clustering K-Means untuk mengelompokkan perjalanan, dan menghasilkan laporan interaktif dalam [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) untuk optimasi rute, perencanaan perawatan, deteksi anomali, dan peningkatan efisiensi.

**Cara Menjalankan**:
1. Pastikan kredensial Snowflake di [`vehicle_tracking.py`](./vehicle_tracking.py) sudah benar.
2. Jalankan skrip:
   ```bash
   python vehicle_tracking.py
   ```
3. Program akan:
   - Mengambil data dari tabel `tr_track` di Snowflake.
   - Memproses data (konversi durasi, ekstraksi jarak, pengkodean kategorikal).
   - Menerapkan K-Means dengan metode Elbow untuk menentukan jumlah klaster optimal (default: 4 klaster).
   - Menghasilkan file [`vehicle_tracking_report.html`](./vehicle_tracking_report.html).
4. Buka [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) di browser untuk melihat laporan interaktif dengan grafik dan tabel.

**Catatan**:
- Pastikan koneksi internet stabil untuk mengakses Snowflake.
- File [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) menggunakan Chart.js dan Tailwind CSS via CDN, sehingga memerlukan internet untuk rendering penuh. File ini dihasilkan oleh [`vehicle_tracking.py`](./vehicle_tracking.py) dan berisi laporan visual seperti grafik bar, kurva Elbow, dan tabel ringkasan.
- Untuk melihat isi [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) tanpa menjalankan skrip, buka file tersebut langsung di browser, tetapi data mungkin tidak mencerminkan hasil terbaru dari Snowflake.

---

## Hasil

### Hasil Soal 1
- **Output**: GUI menampilkan video dengan:
  - Bounding box di sekitar objek yang terdeteksi, berwarna sesuai kelas (`Ripe`: hijau, `Unripe`: kuning, dll.).
  - ID pelacakan dan kecepatan vertikal (vy) pada setiap bounding box.
  - Garis penghitung merah pada posisi Y yang ditentukan.
  - Hitungan kelas di panel kanan, diperbarui saat objek melewati garis penghitung.
- **Video Hasil**:
  - [**Processed Video (Inference Every Frame)**](https://youtu.be/uu9BNt9p-cM):
    - Pratinjau (klik untuk memutar di YouTube):  
      [![](https://img.youtube.com/vi/uu9BNt9p-cM/0.jpg)](https://youtu.be/uu9BNt9p-cM)
    - Video Player Lokal: Lihat di [Video Player HTML](./video_player.html) untuk pemutaran file lokal (hanya berfungsi di platform yang mendukung HTML, seperti GitHub Pages).
    - Deskripsi: Menunjukkan hasil inference dengan pemrosesan setiap frame, memberikan pelacakan yang lebih halus dan akurat. [Unduh file lokal](./video/precessed-1-frame.mp4).
    <!-- Untuk platform yang mendukung HTML (tidak berfungsi di GitHub) -->
    <iframe width="320" height="180" src="https://www.youtube.com/embed/uu9BNt9p-cM" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  - [**Processed Video (Inference Every 30 Frames)**](https://youtu.be/KJyNu4lUbdk):
    - Pratinjau (klik untuk memutar di YouTube):  
      [![](https://img.youtube.com/vi/KJyNu4lUbdk/0.jpg)](https://youtu.be/KJyNu4lUbdk)
    - Video Player Lokal: Lihat di [Video Player HTML](./video_player.html) untuk pemutaran file lokal (hanya berfungsi di platform yang mendukung HTML, seperti GitHub Pages).
    - Deskripsi: Menunjukkan hasil inference sesuai spesifikasi awal (setiap 30 frame, ~1 detik pada 30 FPS), yang mungkin memiliki pelacakan kurang akurat tetapi lebih hemat sumber daya. [Unduh file lokal](./video/precessed-30-frame.mp4).
    <!-- Untuk platform yang mendukung HTML (tidak berfungsi di GitHub) -->
    <iframe width="320" height="180" src="https://www.youtube.com/embed/KJyNu4lUbdk" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
- **Performa**:
  - Akurasi deteksi bergantung pada model YOLO ([`22K-5-M.pt`](./model/22K-5-M.pt)). Berdasarkan [`analisis.md`](./analisis.md), model memiliki mAP@50 55,4%, dengan performa baik pada Kelas 0 (`Ripe`) tetapi buruk pada Kelas 3 (`Rotten`).
  - Pelacakan stabil menggunakan Kalman Filter, dengan penanganan objek yang hilang hingga 60 frame.
  - FPS bervariasi (tergantung perangkat keras), ditampilkan di GUI.
- **Fitur Tambahan**:
  - Penyesuaian kecepatan vertikal (`fastest_vy`) secara dinamis berdasarkan deteksi.
  - Pemeriksaan tumpang tindih jalur untuk menghindari duplikasi pelacakan.
  - GUI responsif dengan thumbnail video, progress bar, dan kontrol interaktif.

### Hasil Soal 2
- **Output**: Dokumen [`analisis.md`](./analisis.md) berisi:
  - **Distribusi Kelas**: Dataset tidak seimbang, memengaruhi performa kelas minoritas (Kelas 3 dan 4).
  - **Metrik Evaluasi**:
    - mAP@50: 55,4% (sedang).
    - Precision: 54,5%, Recall: 54,0% (seimbang tetapi perlu perbaikan).
    - mAP@50 per kelas: Kelas 0 (91,6-92,5%), Kelas 3 (8,7-8,8%).
  - **Analisis Loss**:
    - Loss pelatihan (box, class, DFL) menurun konsisten.
    - Loss validasi menurun lebih lambat, menunjukkan overfitting.
  - **Interpretasi**:
    - Model baik untuk Kelas 0, buruk untuk Kelas 3 dan 4 karena data terbatas.
    - Rekomendasi: Oversampling, regularisasi, dan tambahan data untuk kelas minoritas.
- **Visualisasi**: Grafik dalam [`train_result/`](./train_result/) (misalnya, [`Training_Graphs.png`](./train_result/Training_Graphs.png)) mendukung analisis dengan kurva loss dan metrik.

### Hasil Soal 3
- **Output**: File [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) (dihasilkan oleh [`vehicle_tracking.py`](./vehicle_tracking.py)) berisi laporan interaktif dengan:
  - **Ringkasan Data**:
    - Jumlah peristiwa: 103.046
    - Rata-rata durasi: 5.251 detik
    - Rata-rata jarak: 4,19 km
  - **Optimasi Rute**:
    - Lokasi sering dikunjungi: `PT_2GHTI26CT2` (6.324 kunjungan), `PT_JSJ919` (5.496 kunjungan), `MILL` (3.524 kunjungan).
    - Klaster tidak efisien (Kelompok 0 dan 3) dengan efisiensi rendah (1,08 dan 0,00 km/jam).
  - **Perencanaan Perawatan**:
    - Identifikasi kendaraan dengan perjalanan jauh untuk perawatan tepat waktu.
  - **Deteksi Anomali**:
    - Contoh: Kendaraan `O5422DT028` di Klaster 3 dengan durasi 8.021 menit dan jarak 0 km (kemungkinan parkir tidak normal).
  - **Peningkatan Efisiensi**:
    - Rekomendasi rute alternatif untuk klaster tidak efisien.
  - **Kurva Elbow**: Menunjukkan 4 klaster sebagai jumlah optimal.
  - **Ringkasan Klaster**:
    - Klaster 0: Perjalanan Tidak Efisien (70.407 peristiwa, efisiensi 1,08 km/jam).
    - Klaster 2: Perjalanan Reguler (27.898 peristiwa, efisiensi 16,51 km/jam).
    - Klaster 3: Perjalanan Tidak Efisien (581 peristiwa, efisiensi 0,00 km/jam).
    - Klaster 1: Perjalanan Reguler (4.160 peristiwa, efisiensi 22,85 km/jam).
- **Performa**:
  - Clustering K-Means efektif mengelompokkan perjalanan berdasarkan durasi, jarak, efisiensi, dan lokasi.
  - Deteksi anomali menggunakan threshold jarak ke pusat klaster (top 5% sebagai anomali).
  - Laporan HTML interaktif dengan grafik bar dan kurva Elbow menggunakan Chart.js, tersedia dalam [`vehicle_tracking_report.html`](./vehicle_tracking_report.html).
- **Referensi ke vehicle_tracking_report.html**:
  - File ini adalah output utama dari [`vehicle_tracking.py`](./vehicle_tracking.py) dan berisi visualisasi serta ringkasan data dalam format HTML interaktif.
  - Untuk melihat laporan, jalankan [`vehicle_tracking.py`](./vehicle_tracking.py) untuk menghasilkan file terbaru, lalu buka [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) di browser.
  - Contoh isi laporan dapat dilihat langsung di file [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) yang disediakan, tetapi data mungkin tidak real-time kecuali skrip dijalankan kembali.

---

## Catatan Tambahan
- **Soal 1**:
  - Untuk memenuhi spesifikasi inference setiap 30 frame, ubah `process_frame_interval = 1` menjadi `process_frame_interval = 30` di [`inference_yolo.py`](./inference_yolo.py), tetapi ini dapat mengurangi akurasi pelacakan. Video [`precessed-30-frame.mp4`](./video/precessed-30-frame.mp4) menunjukkan hasil dengan pengaturan ini.
  - Model YOLO rentan terhadap performa buruk pada Kelas 3 (`Rotten`) berdasarkan [`analisis.md`](./analisis.md), sehingga hasil deteksi untuk kelas ini mungkin tidak akurat.
- **Soal 2**:
  - Analisis dapat diperluas dengan metrik tambahan (misalnya, F1-Score) jika data tersedia.
  - Gambar dalam [`train_result/`](./train_result/) harus tersedia untuk validasi analisis.
- **Soal 3**:
  - Kredensial Snowflake harus divalidasi sebelum menjalankan [`vehicle_tracking.py`](./vehicle_tracking.py).
  - Jumlah klaster (4) dipilih berdasarkan kurva Elbow, tetapi dapat disesuaikan dengan inspeksi visual atau metode otomatis (misalnya, Silhouette Score).
  - File [`vehicle_tracking_report.html`](./vehicle_tracking_report.html) bergantung pada CDN untuk Chart.js dan Tailwind CSS. Untuk penggunaan offline, unduh dependensi dan sesuaikan `<script>` dan `<link>` di file tersebut.

Untuk pertanyaan atau bantuan lebih lanjut, silakan hubungi pengelola proyek.