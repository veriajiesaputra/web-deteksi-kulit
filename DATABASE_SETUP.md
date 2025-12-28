# Setup Database MySQL untuk Skinalyze

## Persyaratan
- MySQL Server (versi 5.7 atau lebih baru)
- Python 3.8+
- pip

## Langkah-langkah Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup MySQL Database

#### Opsi A: Menggunakan SQL File
```bash
mysql -u root -p < database_setup.sql
```

#### Opsi B: Manual Setup
1. Login ke MySQL:
```bash
mysql -u root -p
```

2. Jalankan perintah berikut:
```sql
CREATE DATABASE IF NOT EXISTS skinalyze_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE skinalyze_db;
```

3. Tabel akan dibuat otomatis saat pertama kali menjalankan aplikasi.

### 3. Konfigurasi Database

Edit file `config.py` atau set environment variables:

```bash
# Windows (PowerShell)
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="password_anda"
$env:MYSQL_DATABASE="skinalyze_db"
$env:SECRET_KEY="ubah-dengan-secret-key-yang-aman"

# Linux/Mac
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_USER="root"
export MYSQL_PASSWORD="password_anda"
export MYSQL_DATABASE="skinalyze_db"
export SECRET_KEY="ubah-dengan-secret-key-yang-aman"
```

Atau edit langsung di `config.py`:
```python
MYSQL_HOST = 'localhost'
MYSQL_PORT = 3306
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'password_anda'
MYSQL_DATABASE = 'skinalyze_db'
SECRET_KEY = 'ubah-dengan-secret-key-yang-aman'
```

### 4. Jalankan Aplikasi
```bash
python app.py
```

Tabel `users` akan dibuat otomatis saat pertama kali menjalankan aplikasi.

## Struktur Database

### Tabel: users
- `id` (INT, PRIMARY KEY, AUTO_INCREMENT)
- `username` (VARCHAR(80), UNIQUE, NOT NULL)
- `email` (VARCHAR(120), UNIQUE, NOT NULL)
- `password_hash` (VARCHAR(255), NOT NULL)
- `full_name` (VARCHAR(100), NULL)
- `phone` (VARCHAR(20), NULL)
- `created_at` (DATETIME, DEFAULT CURRENT_TIMESTAMP)
- `updated_at` (DATETIME, DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP)

## Troubleshooting

### Error: "Access denied for user"
- Pastikan username dan password MySQL benar
- Pastikan user memiliki permission untuk membuat database dan tabel

### Error: "Can't connect to MySQL server"
- Pastikan MySQL server sedang berjalan
- Cek host dan port di config.py

### Error: "Unknown database"
- Pastikan database sudah dibuat
- Atau biarkan aplikasi membuat database otomatis

## Catatan Keamanan

1. **Jangan commit** file `config.py` dengan password asli ke repository
2. Gunakan environment variables untuk production
3. Ubah `SECRET_KEY` dengan string random yang aman
4. Pastikan password MySQL kuat dan aman


