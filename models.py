from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    role = db.Column(db.String(20), nullable=False, default='user')  # 'admin' or 'user'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def set_password(self, password):
        """Hash password dan simpan"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Cek apakah password benar"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def to_dict(self):
        """Convert user ke dictionary untuk JSON"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'phone': self.phone,
            'role': self.role,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(255), nullable=True)  # Path ke gambar yang diupload
    image_base64 = db.Column(db.Text, nullable=True)  # Base64 image untuk preview
    all_probabilities = db.Column(db.Text, nullable=True)  # JSON string untuk semua probabilitas
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))
    
    def to_dict(self):
        """Convert prediction history ke dictionary untuk JSON"""
        import json
        return {
            'id': self.id,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'image_base64': self.image_base64,
            'all_probabilities': json.loads(self.all_probabilities) if self.all_probabilities else {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'created_at_formatted': self.created_at.strftime('%d %B %Y, %H:%M') if self.created_at else '-'
        }
    
    def __repr__(self):
        return f'<PredictionHistory {self.id} - {self.predicted_class}>'

