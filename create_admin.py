"""
Script untuk membuat user admin pertama
Jalankan: python create_admin.py
"""
from app import app
from models import db, User

def create_admin():
    with app.app_context():
        # Cek apakah sudah ada admin
        admin_exists = User.query.filter_by(role='admin').first()
        if admin_exists:
            print("⚠️  Admin sudah ada. Gunakan user yang sudah ada atau ubah role user yang ada.")
            return
        
        print("=" * 50)
        print("Membuat Admin User")
        print("=" * 50)
        
        username = input("Username: ").strip()
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        full_name = input("Full Name (optional): ").strip() or None
        
        if not username or not email or not password:
            print("❌ Username, email, dan password harus diisi!")
            return
        
        # Cek apakah username atau email sudah ada
        if User.query.filter_by(username=username).first():
            print(f"❌ Username '{username}' sudah digunakan!")
            return
        
        if User.query.filter_by(email=email).first():
            print(f"❌ Email '{email}' sudah terdaftar!")
            return
        
        # Buat admin user
        try:
            admin = User(
                username=username,
                email=email,
                full_name=full_name,
                role='admin'
            )
            admin.set_password(password)
            
            db.session.add(admin)
            db.session.commit()
            
            print("\n✅ Admin user berhasil dibuat!")
            print(f"   Username: {username}")
            print(f"   Email: {email}")
            print(f"   Role: admin")
            print("\n🌐 Login di: http://127.0.0.1:5000/login")
            print("📊 Admin panel: http://127.0.0.1:5000/admin")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error: {e}")

if __name__ == '__main__':
    create_admin()


