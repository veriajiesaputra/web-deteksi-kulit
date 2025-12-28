import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'kuncirahasia'
    
    # Ambil link database dari Railway
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url:
        # PENTING: Ubah mysql:// jadi mysql+pymysql:// agar tidak error di Python
        if database_url.startswith("mysql://"):
            database_url = database_url.replace("mysql://", "mysql+pymysql://", 1)
        SQLALCHEMY_DATABASE_URI = database_url
    else:
        # Jika dijalankan di laptop (lokal), otomatis pakai SQLite
        SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
        
    SQLALCHEMY_TRACK_MODIFICATIONS = False