# Admin Panel Setup Guide

## Overview
Panel admin terpisah untuk mengelola users, predictions, dan melihat statistik dashboard.

## Setup Database

1. **Update database schema** (jika sudah ada database):
   ```sql
   ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user';
   ```

   Atau jalankan ulang `database_setup.sql` yang sudah diupdate.

## Membuat Admin User

### Cara 1: Menggunakan Script
```bash
python create_admin.py
```

Ikuti instruksi untuk memasukkan:
- Username
- Email
- Password
- Full Name (optional)

### Cara 2: Manual via MySQL
```sql
USE skinalyze_db;

-- Ganti dengan username, email, dan password hash yang sesuai
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@example.com', '<password_hash>', 'admin');
```

**Note:** Untuk password hash, gunakan Python:
```python
from werkzeug.security import generate_password_hash
print(generate_password_hash('your_password'))
```

### Cara 3: Ubah Role User yang Sudah Ada
Login sebagai user biasa, lalu di database:
```sql
UPDATE users SET role = 'admin' WHERE username = 'your_username';
```

## Mengakses Admin Panel

1. Login dengan akun admin di: `http://127.0.0.1:5000/login`
2. Klik link "Admin" di navbar (hanya muncul untuk admin)
3. Atau langsung akses: `http://127.0.0.1:5000/admin`

## Fitur Admin Panel

### Dashboard (`/admin`)
- **Total Users**: Jumlah total user, dengan statistik user baru minggu ini
- **Total Predictions**: Jumlah total prediksi, dengan statistik prediksi minggu ini
- **Predictions Today**: Prediksi hari ini dengan persentase
- **Regular Users**: Jumlah user biasa dan admin
- **Top Predicted Classes**: 5 kelas penyakit yang paling sering diprediksi
- **Recent Predictions**: 10 prediksi terbaru dengan detail user

### User Management (`/admin/users`)
- **List semua users** dengan search dan filter role
- **View user details** dengan history prediksi per user
- **Toggle role** user (admin/user)
- **Delete user** (akan menghapus semua data user termasuk predictions)

### User Detail (`/admin/users/<user_id>`)
- Informasi lengkap user
- Statistik prediksi user
- History prediksi user dengan pagination
- Hapus prediksi individual

### All Predictions (`/admin/predictions`)
- **List semua predictions** dari semua user
- **Search** berdasarkan user, email, atau prediction class
- **Filter** berdasarkan prediction class
- **View user** dari prediksi
- **Delete prediction**

## Security

- Semua routes admin dilindungi dengan `@admin_required` decorator
- Hanya user dengan `role='admin'` yang bisa mengakses
- User biasa akan di-redirect ke homepage jika mencoba akses admin panel

## File Structure

```
templates/admin/
├── base.html          # Base template dengan sidebar
├── dashboard.html     # Dashboard dengan statistik
├── users.html         # User management
├── user_detail.html   # Detail user dan history
└── predictions.html   # Semua predictions

static/
├── css/
│   └── admin.css      # Styles untuk admin panel
└── js/
    └── admin.js       # JavaScript untuk admin panel
```

## Troubleshooting

### Admin panel tidak muncul di navbar
- Pastikan user sudah login dan memiliki `role='admin'`
- Cek di database: `SELECT username, role FROM users WHERE username='your_username';`

### Tidak bisa akses `/admin`
- Pastikan sudah login sebagai admin
- Cek console untuk error messages
- Pastikan decorator `@admin_required` sudah terpasang di routes

### Database error saat membuat admin
- Pastikan kolom `role` sudah ada di tabel `users`
- Jalankan migration SQL jika perlu

