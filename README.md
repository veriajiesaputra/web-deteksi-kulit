# Sistem Klasifikasi Penyakit Kulit dengan Deep Learning

## 📋 Daftar Isi
1. [Latar Belakang](#latar-belakang)
2. [Tujuan](#tujuan)
3. [Pemahaman Data](#pemahaman-data)
4. [Processing Data](#processing-data)
5. [Metode yang Digunakan](#metode-yang-digunakan)
6. [Gambaran Proyek](#gambaran-proyek)
7. [Deployment](#deployment)
8. [Cara Penggunaan](#cara-penggunaan)
9. [Struktur Proyek](#struktur-proyek)

---

## 🎯 Latar Belakang

Penyakit kulit merupakan salah satu masalah kesehatan yang sering terjadi di masyarakat. Deteksi dini penyakit kulit sangat penting untuk mencegah komplikasi lebih lanjut dan memungkinkan pengobatan yang lebih efektif. Namun, diagnosis penyakit kulit secara manual memerlukan keahlian medis khusus dan dapat memakan waktu yang lama.

Dengan perkembangan teknologi **Deep Learning** dan **Computer Vision**, sistem klasifikasi penyakit kulit berbasis AI dapat membantu dalam proses deteksi awal. Sistem ini dapat menganalisis gambar lesi kulit dan memberikan prediksi jenis penyakit dengan akurasi yang tinggi, sehingga dapat menjadi alat bantu untuk tenaga medis maupun masyarakat umum dalam melakukan screening awal.

**Masalah yang Dihadapi:**
- Keterbatasan akses ke dokter spesialis kulit
- Waktu tunggu yang lama untuk konsultasi
- Biaya konsultasi yang relatif mahal
- Kebutuhan akan alat bantu diagnosis yang cepat dan akurat

**Solusi yang Ditawarkan:**
- Sistem klasifikasi otomatis menggunakan Deep Learning
- Prediksi cepat dalam hitungan detik
- Akurasi tinggi dengan model yang telah dilatih
- Akses mudah melalui aplikasi web

---

## 🎯 Tujuan

### Tujuan Umum
Mengembangkan sistem klasifikasi penyakit kulit berbasis Deep Learning yang dapat mengidentifikasi berbagai jenis penyakit kulit dari gambar dengan akurasi tinggi.

### Tujuan Khusus
1. **Membangun Model Deep Learning**
   - Mengembangkan model klasifikasi untuk 9 jenis penyakit kulit
   - Mencapai akurasi validasi di atas 85%
   - Menggunakan transfer learning untuk efisiensi training

2. **Mengembangkan Aplikasi Web**
   - Membuat interface yang user-friendly
   - Menyediakan fitur upload gambar dan prediksi
   - Menampilkan hasil prediksi dengan confidence score

3. **Menyediakan Informasi Edukasi**
   - Menyediakan artikel tentang berbagai penyakit kulit
   - Menampilkan contoh gambar dari dataset
   - Memberikan informasi pengobatan dan pencegahan

4. **Deployment Aplikasi**
   - Mengembangkan aplikasi web dengan Flask
   - Membuat sistem yang dapat diakses secara online
   - Memastikan performa dan kehandalan sistem

---

## 📊 Pemahaman Data

### Dataset
Dataset yang digunakan terdiri dari **9 kelas penyakit kulit** dengan total **9.888 gambar**:

| No | Kelas Penyakit | Jumlah Gambar |
|---|----------------|---------------|
| 1 | Actinic keratosis | 1.137 |
| 2 | Atopic Dermatitis | 1.136 |
| 3 | Benign keratosis | 1.143 |
| 4 | Dermatofibroma | 1.142 |
| 5 | Melanocytic nevus | 1.129 |
| 6 | Melanoma | 1.125 |
| 7 | Squamous cell carcinoma | 1.132 |
| 8 | Tinea Ringworm Candidiasis | 810 |
| 9 | Vascular lesion | 1.134 |

### Karakteristik Dataset
- **Format**: JPG/JPEG
- **Ukuran**: Bervariasi (akan di-resize menjadi 224x224)
- **Distribusi**: Relatif seimbang (kecuali Tinea Ringworm Candidiasis)
- **Kualitas**: Gambar dengan resolusi yang baik
- **Sumber**: Dataset medis terpercaya

### Analisis Data
- **Total gambar**: 9.888
- **Rata-rata per kelas**: ~1.099 gambar
- **Kelas dengan gambar terbanyak**: Benign keratosis (1.143)
- **Kelas dengan gambar tersedikit**: Tinea Ringworm Candidiasis (810)
- **Imbalance ratio**: 1.4:1 (masih dalam batas wajar)

### Preprocessing yang Diperlukan
- Resize gambar ke ukuran seragam (224x224)
- Normalisasi pixel values (0-1)
- Augmentasi data untuk meningkatkan generalisasi
- Split data menjadi training dan validation set

---

## 🔄 Processing Data

### 1. Data Loading
```python
# Menggunakan ImageDataGenerator dari Keras
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15
)
```

### 2. Data Augmentation
Augmentasi data diterapkan pada training set untuk meningkatkan variasi data dan mencegah overfitting:

- **Rotation Range**: 20° (rotasi gambar)
- **Zoom Range**: 0.15 (zoom in/out hingga 15%)
- **Width Shift**: 0.1 (geser horizontal 10%)
- **Height Shift**: 0.1 (geser vertikal 10%)
- **Horizontal Flip**: True (flip horizontal)

**Alasan Augmentasi:**
- Meningkatkan jumlah data training secara virtual
- Meningkatkan generalisasi model
- Mengurangi overfitting
- Menangani variasi orientasi dan posisi gambar

### 3. Data Splitting
- **Training Set**: 85% (8.409 gambar)
- **Validation Set**: 15% (1.479 gambar)
- **Metode**: Stratified split untuk menjaga proporsi kelas

### 4. Normalisasi
- **Rescale**: 1./255 (mengubah nilai pixel dari 0-255 menjadi 0-1)
- **Alasan**: Mempercepat konvergensi training dan meningkatkan stabilitas

### 5. Batch Processing
- **Batch Size**: 32
- **Target Size**: 224x224 pixels
- **Color Mode**: RGB (3 channels)

---

## 🧠 Metode yang Digunakan

### 1. Transfer Learning dengan MobileNetV2

**MobileNetV2** dipilih sebagai base model karena:
- **Efisien**: Model ringan dan cepat
- **Pre-trained**: Sudah dilatih pada ImageNet (1.4M gambar, 1000 kelas)
- **Depthwise Separable Convolutions**: Mengurangi parameter dan komputasi
- **Inverted Residuals**: Meningkatkan representasi fitur

### 2. Arsitektur Model

```
Input Layer (224x224x3)
    ↓
MobileNetV2 Base (frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.2)
    ↓
Dense (128, ReLU)
    ↓
Dropout (0.2)
    ↓
Dense (9, Softmax) → Output
```

**Komponen Model:**
- **Base Model**: MobileNetV2 (frozen, tidak di-train)
- **GlobalAveragePooling2D**: Mengurangi dimensi feature map
- **Dropout**: Mencegah overfitting (0.2)
- **Dense Layer**: 128 neurons dengan ReLU activation
- **Output Layer**: 9 neurons (sesuai jumlah kelas) dengan Softmax

### 3. Hyperparameters

| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| Learning Rate | 1e-4 | Learning rate kecil untuk fine-tuning |
| Optimizer | Adam | Adaptive learning rate |
| Batch Size | 32 | Balance antara memori dan stabilitas |
| Epochs | 12 | Dengan early stopping |
| Loss Function | Categorical Crossentropy | Multi-class classification |
| Metrics | Accuracy | Mengukur performa model |

### 4. Callbacks

- **EarlyStopping**: 
  - Monitor: `val_loss`
  - Patience: 3
  - Menghentikan training jika tidak ada perbaikan

- **ModelCheckpoint**:
  - Monitor: `val_accuracy`
  - Save best only: True
  - Menyimpan model terbaik berdasarkan validation accuracy

### 5. Hasil Training

- **Best Validation Accuracy**: 87.09%
- **Best Validation Loss**: 0.3680
- **Training Time**: ~12-15 menit per epoch (CPU)
- **Model Size**: ~9 MB (MobileNetV2)

---

## 🚀 Gambaran Proyek

### 1. Development Phase

#### a. Data Preparation
- Mengorganisir dataset ke dalam folder per kelas
- Memeriksa kualitas dan konsistensi gambar
- Menyiapkan struktur folder untuk training

#### b. Model Development (Jupyter Notebook)
- **Exploratory Data Analysis (EDA)**
  - Analisis distribusi kelas
  - Visualisasi contoh gambar
  - Pemeriksaan ukuran gambar

- **Data Preprocessing**
  - Setup ImageDataGenerator
  - Konfigurasi augmentasi
  - Split data training/validation

- **Model Building**
  - Load MobileNetV2 pre-trained
  - Build custom classifier head
  - Compile model dengan optimizer dan loss function

- **Training**
  - Training dengan callbacks
  - Monitoring training history
  - Visualisasi accuracy dan loss

- **Evaluation**
  - Prediksi pada validation set
  - Confusion matrix
  - Classification report (precision, recall, F1-score)

- **Model Saving**
  - Simpan model terbaik sebagai `skin_model.h5`
  - Simpan class indices sebagai `class_indices.json`

#### c. Web Application Development

**Backend (Flask):**
- API endpoint untuk prediksi (`/api/predict`)
- API endpoint untuk artikel (`/api/diseases`, `/api/disease/<name>`)
- API endpoint untuk contoh gambar (`/api/disease/<name>/images`)
- Image preprocessing untuk inference
- Model loading dan caching

**Frontend:**
- **Halaman Beranda** (`index.html`)
  - Hero section
  - Fitur utama
  - Preview artikel penyakit

- **Halaman Prediksi** (`predict.html`)
  - Upload gambar (drag & drop)
  - Preview gambar
  - Hasil prediksi dengan confidence score
  - Probabilitas semua kelas (≥10%)

- **Halaman Artikel** (`artikel.html`)
  - Dropdown pemilihan penyakit
  - Informasi lengkap penyakit
  - Contoh gambar dari dataset
  - Artikel edukasi

**Styling (CSS):**
- Modern UI design
- Responsive layout
- Dark mode support
- Animasi dan transisi

### 2. Features Implemented

✅ **Upload & Prediction**
- Drag & drop file upload
- Preview gambar sebelum prediksi
- Prediksi dengan confidence score
- Probabilitas semua kelas (filtered ≥10%)

✅ **Artikel Edukasi**
- 9 artikel penyakit kulit lengkap
- Penjelasan, gejala, dan pengobatan
- Contoh gambar dari dataset
- Auto-select dari URL parameter

✅ **User Experience**
- Dark mode toggle
- Responsive design (mobile-friendly)
- Loading indicators
- Error handling
- Smooth animations

### 3. Project Structure

```
projectskindisease/
├── app.py                          # Flask application
├── skin_model.h5                   # Trained model
├── class_indices.json              # Class mapping
├── requirements.txt                # Dependencies
├── README.md                       # Documentation
├── skin_disease_classification.ipynb  # Jupyter notebook
├── dataset/                        # Dataset folder
│   ├── Actinic keratosis/
│   ├── Atopic Dermatitis/
│   ├── Benign keratosis/
│   ├── Dermatofibroma/
│   ├── Melanocytic nevus/
│   ├── Melanoma/
│   ├── Squamous cell carcinoma/
│   ├── Tinea Ringworm Candidiasis/
│   └── Vascular lesion/
├── templates/                      # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   └── artikel.html
└── static/                         # Static files
    ├── css/
    │   └── styles.css
    └── js/
        └── main.js
```

---

## 🌐 Deployment

### Teknologi yang Digunakan untuk Deployment

#### **1. Flask (Web Framework)**

**Flask** digunakan sebagai web framework untuk membangun aplikasi karena:
- **Ringan dan Fleksibel**: Framework minimalis yang mudah dikustomisasi
- **Python-based**: Konsisten dengan stack teknologi (TensorFlow/Keras)
- **RESTful API**: Mudah membuat API endpoints untuk prediksi
- **Template Engine**: Jinja2 untuk rendering HTML templates
- **Static Files**: Built-in support untuk CSS, JavaScript, dan images

**Fitur Flask yang Digunakan:**
- Route handlers untuk halaman web (`@app.route`)
- File upload handling (`request.files`)
- JSON responses untuk API (`jsonify`)
- Template rendering (`render_template`)
- Static file serving

#### **2. TensorFlow/Keras (Deep Learning)**

**TensorFlow/Keras** digunakan untuk:
- **Model Loading**: Load model yang sudah di-train (`load_model`)
- **Inference**: Prediksi pada gambar baru
- **Preprocessing**: Normalisasi dan resize gambar
- **Model Format**: Model disimpan sebagai `.h5` file

**Alasan Penggunaan:**
- Model sudah di-train dengan TensorFlow/Keras
- Konsisten dengan training pipeline
- Optimized untuk inference
- Support untuk transfer learning models

#### **3. PIL/Pillow (Image Processing)**

**Pillow** digunakan untuk:
- **Image Loading**: Load gambar dari file upload
- **Image Resize**: Resize ke ukuran yang dibutuhkan (224x224)
- **Format Conversion**: Convert ke RGB format
- **Image Preview**: Generate preview untuk ditampilkan

#### **4. HTML/CSS/JavaScript (Frontend)**

**Frontend Technologies:**
- **HTML5**: Struktur halaman web
- **CSS3**: Styling dan layout (responsive design, dark mode)
- **JavaScript**: Interaktivitas (upload, API calls, dynamic content)
- **Fetch API**: Komunikasi dengan backend API

#### **5. File System (Storage)**

**Storage yang Digunakan:**
- **Local File System**: Menyimpan model dan dataset
- **Model File**: `skin_model.h5` (~9 MB)
- **Class Mapping**: `class_indices.json`
- **Dataset Images**: Folder `dataset/` untuk contoh gambar

### Cara Menjalankan Aplikasi

**Langkah-langkah:**

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Pastikan File Model Ada**
- `skin_model.h5` - Model yang sudah di-train
- `class_indices.json` - Mapping kelas penyakit

3. **Jalankan Aplikasi**
```bash
python app.py
```

4. **Akses Aplikasi**
```
http://localhost:5000
```

### Spesifikasi Sistem

- **Python Version**: 3.8+
- **Framework**: Flask 3.0.0
- **Deep Learning**: TensorFlow 2.15.0
- **Image Processing**: Pillow 10.1.0
- **Model Size**: ~9 MB
- **Inference Time**: ~0.5-1 detik per gambar
- **Memory**: ~500 MB - 1 GB RAM

---

## 🌐 Isi Website

### Halaman Beranda (Homepage)

**URL**: `/` atau `/index`

**Fitur:**
1. **Hero Section**
   - Judul: "Deteksi Penyakit Kulit dengan AI"
   - Deskripsi singkat tentang aplikasi
   - Tombol CTA: "Mulai Prediksi"

2. **Fitur Utama (Features Section)**
   - 📸 **Upload Gambar**: Upload foto dengan drag & drop
   - ⚡ **Prediksi Cepat**: Hasil dalam hitungan detik
   - 🧠 **AI Powered**: Ditenagai deep learning
   - 📊 **Hasil Detail**: Confidence score dan probabilitas

3. **Artikel Preview (Articles Preview)**
   - Grid 3 kolom menampilkan 9 artikel penyakit kulit
   - Setiap card menampilkan:
     - Icon penyakit
     - Nama penyakit
     - Deskripsi singkat
     - Link "Baca Selengkapnya"
   - Klik card langsung ke artikel penyakit yang dipilih
   - Tombol "Lihat Semua Artikel"

4. **Call-to-Action Section**
   - Ajakan untuk mulai menggunakan aplikasi
   - Tombol "Mulai Prediksi Sekarang"

### Halaman Prediksi (Prediction Page)

**URL**: `/predict`

**Fitur:**
1. **Upload Section**
   - Area drag & drop untuk upload gambar
   - Atau klik untuk memilih file
   - Format: JPG, JPEG, PNG (max 16MB)
   - Preview gambar sebelum prediksi
   - Tombol "Prediksi Sekarang"

2. **Hasil Prediksi (Result Section)**
   - **Layout Presisi**:
     - **Baris 1**: Gambar | Kelas Prediksi (berjejer)
     - **Baris 2**: Probabilitas (≥10%) di bawah
   
   - **Gambar Hasil**:
     - Preview gambar yang diupload
     - Ukuran maksimal 400px
     - Rounded corners dengan shadow
   
   - **Kelas Prediksi Card**:
     - Background gradient (biru-ungu)
     - Nama kelas prediksi (besar, bold)
     - Confidence score (persentase)
     - Design card yang menarik
   
   - **Probabilitas Card**:
     - Menampilkan semua kelas dengan probabilitas ≥10%
     - Bar chart untuk setiap kelas
     - Diurutkan dari tertinggi ke terendah
     - Highlight untuk kelas yang diprediksi

3. **Loading Indicator**
   - Spinner animation saat memproses
   - Pesan "Memproses gambar..."
   - Overlay untuk mencegah interaksi

4. **Error Handling**
   - Pesan error jika upload gagal
   - Validasi format file
   - Validasi ukuran file

5. **Action Buttons**
   - "Unggah Gambar Baru" untuk reset

### Halaman Artikel (Article Page)

**URL**: `/artikel`

**Fitur:**
1. **Dropdown Selection**
   - Dropdown untuk memilih penyakit dari 9 kelas
   - Auto-select dari URL parameter (jika dari card)
   - Loading indicator saat memuat

2. **Informasi Penyakit (Disease Info)**
   - **Header**:
     - Nama penyakit (besar, bold)
   
   - **Penjelasan (📋 Penjelasan)**:
     - Deskripsi lengkap tentang penyakit
     - Penyebab dan karakteristik
     - Paragraf informatif
   
   - **Pengobatan & Perawatan (💊 Pengobatan & Perawatan)**:
     - List pengobatan yang disarankan
     - Tips perawatan
     - Rekomendasi medis
   
   - **Contoh Gambar (🖼️ Contoh Gambar dari Dataset)**:
     - Grid 4 gambar contoh dari dataset
     - Random selection untuk variasi
     - Lazy loading untuk performa
     - Hover effect pada gambar

3. **Navigation**
   - Auto-scroll ke informasi saat penyakit dipilih
   - Smooth scroll animation
   - URL update saat dropdown berubah

### Fitur Umum (Semua Halaman)

1. **Navigation Bar**
   - Logo "SehatinKulit"
   - Menu: Beranda, Artikel, Prediksi
   - Dark mode toggle (🌙/☀️)
   - Sticky navbar (tetap di atas saat scroll)

2. **Dark Mode**
   - Toggle button di navbar
   - Preferensi tersimpan di localStorage
   - Smooth transition
   - Support untuk semua halaman

3. **Responsive Design**
   - Mobile-friendly (breakpoint 768px)
   - Tablet-friendly (breakpoint 1024px)
   - Desktop-optimized
   - Touch-friendly untuk mobile

4. **Footer**
   - Copyright information
   - Simple dan clean design

### API Endpoints

1. **POST `/api/predict`**
   - Upload gambar untuk prediksi
   - Request: Form data dengan file image
   - Response: JSON dengan predicted_class, confidence, all_probabilities, image_preview

2. **GET `/api/diseases`**
   - Daftar semua penyakit
   - Response: JSON dengan list diseases (name, display_name, image_count)

3. **GET `/api/diseases/preview`**
   - Daftar penyakit untuk preview (homepage)
   - Response: JSON dengan list diseases lengkap

4. **GET `/api/disease/<disease_name>`**
   - Informasi penyakit spesifik
   - Response: JSON dengan display_name, explanation, treatment

5. **GET `/api/disease/<disease_name>/images`**
   - Contoh gambar dari dataset
   - Response: JSON dengan list image URLs dan total_images

6. **GET `/dataset/<disease_name>/<filename>`**
   - Serve gambar dari dataset folder
   - Response: Image file

### User Experience Features

1. **Smooth Animations**
   - Hover effects pada cards
   - Transitions pada buttons
   - Loading animations
   - Scroll animations

2. **Visual Feedback**
   - Loading indicators
   - Success states
   - Error messages
   - Hover states

3. **Accessibility**
   - Semantic HTML
   - Alt text untuk images
   - Keyboard navigation
   - Screen reader friendly

4. **Performance**
   - Lazy loading images
   - Optimized CSS/JS
   - Efficient API calls
   - Cached model loading

---

## 🌐 Isi Website

### Halaman Beranda (Homepage)

**URL**: `/` atau `/index`

**Fitur:**
1. **Hero Section**
   - Judul: "Deteksi Penyakit Kulit dengan AI"
   - Deskripsi singkat tentang aplikasi
   - Tombol CTA: "Mulai Prediksi"

2. **Fitur Utama (Features Section)**
   - 📸 **Upload Gambar**: Upload foto dengan drag & drop
   - ⚡ **Prediksi Cepat**: Hasil dalam hitungan detik
   - 🧠 **AI Powered**: Ditenagai deep learning
   - 📊 **Hasil Detail**: Confidence score dan probabilitas

3. **Artikel Preview (Articles Preview)**
   - Grid 3 kolom menampilkan 9 artikel penyakit kulit
   - Setiap card menampilkan:
     - Icon penyakit
     - Nama penyakit
     - Deskripsi singkat
     - Link "Baca Selengkapnya"
   - Klik card langsung ke artikel penyakit yang dipilih
   - Tombol "Lihat Semua Artikel"

4. **Call-to-Action Section**
   - Ajakan untuk mulai menggunakan aplikasi
   - Tombol "Mulai Prediksi Sekarang"

### Halaman Prediksi (Prediction Page)

**URL**: `/predict`

**Fitur:**
1. **Upload Section**
   - Area drag & drop untuk upload gambar
   - Atau klik untuk memilih file
   - Format: JPG, JPEG, PNG (max 16MB)
   - Preview gambar sebelum prediksi
   - Tombol "Prediksi Sekarang"

2. **Hasil Prediksi (Result Section)**
   - **Layout Presisi**:
     - **Baris 1**: Gambar | Kelas Prediksi (berjejer)
     - **Baris 2**: Probabilitas (≥10%) di bawah
   
   - **Gambar Hasil**:
     - Preview gambar yang diupload
     - Ukuran maksimal 400px
     - Rounded corners dengan shadow
   
   - **Kelas Prediksi Card**:
     - Background gradient (biru-ungu)
     - Nama kelas prediksi (besar, bold)
     - Confidence score (persentase)
     - Design card yang menarik
   
   - **Probabilitas Card**:
     - Menampilkan semua kelas dengan probabilitas ≥10%
     - Bar chart untuk setiap kelas
     - Diurutkan dari tertinggi ke terendah
     - Highlight untuk kelas yang diprediksi

3. **Loading Indicator**
   - Spinner animation saat memproses
   - Pesan "Memproses gambar..."
   - Overlay untuk mencegah interaksi

4. **Error Handling**
   - Pesan error jika upload gagal
   - Validasi format file
   - Validasi ukuran file

5. **Action Buttons**
   - "Unggah Gambar Baru" untuk reset

### Halaman Artikel (Article Page)

**URL**: `/artikel`

**Fitur:**
1. **Dropdown Selection**
   - Dropdown untuk memilih penyakit dari 9 kelas
   - Auto-select dari URL parameter (jika dari card)
   - Loading indicator saat memuat

2. **Informasi Penyakit (Disease Info)**
   - **Header**:
     - Nama penyakit (besar, bold)
   
   - **Penjelasan (📋 Penjelasan)**:
     - Deskripsi lengkap tentang penyakit
     - Penyebab dan karakteristik
     - Paragraf informatif
   
   - **Pengobatan & Perawatan (💊 Pengobatan & Perawatan)**:
     - List pengobatan yang disarankan
     - Tips perawatan
     - Rekomendasi medis
   
   - **Contoh Gambar (🖼️ Contoh Gambar dari Dataset)**:
     - Grid 4 gambar contoh dari dataset
     - Random selection untuk variasi
     - Lazy loading untuk performa
     - Hover effect pada gambar

3. **Navigation**
   - Auto-scroll ke informasi saat penyakit dipilih
   - Smooth scroll animation
   - URL update saat dropdown berubah

### Fitur Umum (Semua Halaman)

1. **Navigation Bar**
   - Logo "SehatinKulit"
   - Menu: Beranda, Artikel, Prediksi
   - Dark mode toggle (🌙/☀️)
   - Sticky navbar (tetap di atas saat scroll)

2. **Dark Mode**
   - Toggle button di navbar
   - Preferensi tersimpan di localStorage
   - Smooth transition
   - Support untuk semua halaman

3. **Responsive Design**
   - Mobile-friendly (breakpoint 768px)
   - Tablet-friendly (breakpoint 1024px)
   - Desktop-optimized
   - Touch-friendly untuk mobile

4. **Footer**
   - Copyright information
   - Simple dan clean design

### API Endpoints

1. **POST `/api/predict`**
   - Upload gambar untuk prediksi
   - Request: Form data dengan file image
   - Response: JSON dengan predicted_class, confidence, all_probabilities, image_preview

2. **GET `/api/diseases`**
   - Daftar semua penyakit
   - Response: JSON dengan list diseases (name, display_name, image_count)

3. **GET `/api/diseases/preview`**
   - Daftar penyakit untuk preview (homepage)
   - Response: JSON dengan list diseases lengkap

4. **GET `/api/disease/<disease_name>`**
   - Informasi penyakit spesifik
   - Response: JSON dengan display_name, explanation, treatment

5. **GET `/api/disease/<disease_name>/images`**
   - Contoh gambar dari dataset
   - Response: JSON dengan list image URLs dan total_images

6. **GET `/dataset/<disease_name>/<filename>`**
   - Serve gambar dari dataset folder
   - Response: Image file

### User Experience Features

1. **Smooth Animations**
   - Hover effects pada cards
   - Transitions pada buttons
   - Loading animations
   - Scroll animations

2. **Visual Feedback**
   - Loading indicators
   - Success states
   - Error messages
   - Hover states

3. **Accessibility**
   - Semantic HTML
   - Alt text untuk images
   - Keyboard navigation
   - Screen reader friendly

4. **Performance**
   - Lazy loading images
   - Optimized CSS/JS
   - Efficient API calls
   - Cached model loading

---

## 📖 Cara Penggunaan

### 1. Untuk End User

**Prediksi Penyakit Kulit:**
1. Buka halaman "Prediksi"
2. Upload gambar lesi kulit (JPG/PNG, max 16MB)
3. Klik "Prediksi Sekarang"
4. Lihat hasil prediksi dengan confidence score
5. Lihat probabilitas semua kelas (≥10%)

**Membaca Artikel:**
1. Buka halaman "Artikel"
2. Pilih penyakit dari dropdown
3. Baca penjelasan, gejala, dan pengobatan
4. Lihat contoh gambar dari dataset

### 2. Untuk Developer

**Training Model:**
1. Buka `skin_disease_classification.ipynb`
2. Jalankan semua cells secara berurutan
3. Model akan tersimpan sebagai `skin_model.h5`
4. Class indices tersimpan sebagai `class_indices.json`

**Modifikasi Aplikasi:**
1. Edit `app.py` untuk menambah endpoint
2. Edit templates di folder `templates/`
3. Edit styling di `static/css/styles.css`
4. Restart aplikasi untuk melihat perubahan

---

## 📈 Hasil dan Evaluasi

### Model Performance

**Metrics:**
- **Overall Accuracy**: 87.09%
- **Best Validation Accuracy**: 87.09%
- **Best Validation Loss**: 0.3680

**Per-Class Performance:**
- Actinic keratosis: Precision 0.91, Recall 0.62
- Atopic Dermatitis: Precision 0.97, Recall 0.99
- Benign keratosis: Precision 0.94, Recall 0.96
- Dermatofibroma: Precision 0.85, Recall 0.85
- Melanocytic nevus: Precision 0.86, Recall 0.85
- Melanoma: Precision 0.86, Recall 0.79
- Squamous cell carcinoma: Precision 0.64, Recall 0.91
- Tinea Ringworm Candidiasis: Precision 0.98, Recall 0.94
- Vascular lesion: Precision 0.96, Recall 0.95

### Kelebihan Sistem

✅ **Akurasi Tinggi**: 87% accuracy pada validation set
✅ **Cepat**: Prediksi dalam hitungan detik
✅ **User-Friendly**: Interface yang mudah digunakan
✅ **Edukatif**: Menyediakan informasi lengkap tentang penyakit
✅ **Responsif**: Bekerja dengan baik di berbagai device

### Keterbatasan

⚠️ **Bukan Pengganti Dokter**: Sistem ini hanya alat bantu, bukan pengganti konsultasi medis
⚠️ **Dataset Terbatas**: Hanya 9 kelas penyakit kulit
⚠️ **Kualitas Gambar**: Hasil prediksi bergantung pada kualitas gambar input
⚠️ **False Positives**: Mungkin terjadi kesalahan klasifikasi

---

## 🔮 Pengembangan Selanjutnya

1. **Peningkatan Model**
   - Menambah lebih banyak kelas penyakit
   - Meningkatkan akurasi dengan ensemble methods
   - Fine-tuning dengan dataset yang lebih besar

2. **Fitur Tambahan**
   - History prediksi user
   - Export hasil prediksi
   - Integrasi dengan sistem medis
   - Multi-language support

3. **Optimasi**
   - Model quantization untuk mobile
   - Real-time prediction dengan webcam
   - Batch prediction untuk multiple images

---

## 👥 Kontributor

Proyek ini dikembangkan sebagai bagian dari pembelajaran Deep Learning dan Computer Vision.

---

## 📄 License

Proyek ini dibuat untuk keperluan edukasi dan penelitian.

---

## 🙏 Acknowledgments

- Dataset: Medical skin disease dataset
- Framework: TensorFlow/Keras
- Model: MobileNetV2 (Google)
- Web Framework: Flask

---

**Dibuat dengan ❤️ untuk kesehatan kulit**
