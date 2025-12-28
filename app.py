from flask import Flask, render_template, request, jsonify, url_for, send_from_directory, flash, redirect, session, abort
from jinja2 import Environment
from werkzeug.utils import secure_filename
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
from config import Config
from models import db, User, PredictionHistory
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json
from io import BytesIO
import base64
import random
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import func, desc, inspect
from sqlalchemy.orm import joinedload

app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Silakan login untuk mengakses halaman ini.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Admin decorator
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('Akses ditolak. Hanya admin yang dapat mengakses halaman ini.', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

app.config['UPLOAD_FOLDER'] = 'uploads'
DATA_DIR = "static/dataset"

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model dan class indices saat app start
print("🔄 Loading model...")
MODEL_PATH = "skin_disease_mobilenetv2_stage1.h5"
CLASS_INDICES_PATH = "class_indices.json"

# Cek apakah file model ada
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file tidak ditemukan: {MODEL_PATH}")
    print(f"   Pastikan file model sudah di-train dan disimpan di direktori yang sama dengan app.py")
    model = None
else:
    try:
        model = load_model(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Cek apakah file class indices ada
if not os.path.exists(CLASS_INDICES_PATH):
    print(f"❌ Class indices file tidak ditemukan: {CLASS_INDICES_PATH}")
    print(f"   Pastikan file class_indices.json sudah dibuat setelah training")
    class_indices = {}
    idx_to_class = {}
else:
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # Buat reverse mapping
        idx_to_class = {v: k for k, v in class_indices.items()}
        print(f"✅ Class indices loaded: {len(class_indices)} classes")
    except Exception as e:
        print(f"❌ Error loading class indices: {e}")
        class_indices = {}
        idx_to_class = {}


def allowed_file(filename):
    # Kita tentukan manual di sini biar pasti jalan
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_file, target_size=(224, 224)):
    """
    Preprocess gambar dari file upload
    
    Args:
        image_file: File object dari Flask request.files
        target_size: Ukuran target (default: 224x224)
    
    Returns:
        img_array: Preprocessed image array siap untuk prediksi
    """
    # Load gambar dari file object
    img = Image.open(image_file)
    img = img.convert('RGB')
    img = img.resize(target_size)
    
    # Convert ke array dan normalisasi
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array


def predict_image(image_file):
    """
    Prediksi kelas penyakit kulit dari file upload
    
    Args:
        image_file: File object dari Flask request.files
    
    Returns:
        predicted_class: Nama kelas prediksi
        confidence: Confidence score (probabilitas)
        all_probabilities: Dictionary dengan probabilitas semua kelas
    """
    if model is None:
        raise Exception("Model belum di-load")
    
    # Preprocess
    img_array = preprocess_image(image_file)
    
    # Prediksi
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    
    # Get predicted class
    predicted_class = idx_to_class[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    # Probabilitas semua kelas (diurutkan dari tertinggi)
    all_probabilities = {
        idx_to_class[i]: float(predictions[0][i]) 
        for i in range(len(idx_to_class))
    }
    
    # Sort probabilities
    sorted_probabilities = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
    
    return predicted_class, confidence, sorted_probabilities


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/predict')
@login_required
def predict_page():
    """Page untuk upload gambar dan prediksi - Hanya untuk user yang sudah login"""
    return render_template('predict.html')


@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    """API endpoint untuk prediksi - Hanya untuk user yang sudah login"""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Please upload JPG, JPEG, or PNG'}), 400
        
        # Reset file pointer
        file.seek(0)
        
        # Prediksi
        predicted_class, confidence, all_probabilities = predict_image(file)
        
        
        
        # Convert image to base64 for preview
        file.seek(0)
        img = Image.open(file)
        img = img.convert('RGB')
        
        # Resize untuk preview (max 800px)
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Simpan ke history prediksi
        try:
            history = PredictionHistory(
                user_id=current_user.id,
                predicted_class=predicted_class,
                confidence=confidence,
                image_base64=img_str,
                all_probabilities=json.dumps(all_probabilities)
            )
            db.session.add(history)
            db.session.commit()
        except Exception as e:
            print(f"Error saving prediction history: {e}")
            db.session.rollback()
            # Continue even if history save fails
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'image_preview': f"data:image/jpeg;base64,{img_str}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Data artikel untuk setiap kelas penyakit
DISEASE_INFO = {
    "Actinic keratosis": {
        "display_name": "Actinic Keratosis",
        "explanation": "Actinic keratosis (AK) adalah lesi kulit pra-kanker yang disebabkan oleh paparan sinar matahari jangka panjang. Kondisi ini muncul sebagai bercak kasar, bersisik, atau kerak pada kulit yang sering terpapar sinar UV, terutama pada wajah, telinga, leher, lengan, dan punggung tangan.",
        "treatment": [
            "Krim topikal seperti 5-fluorouracil (5-FU) atau imiquimod",
            "Terapi cryotherapy (pembekuan dengan nitrogen cair)",
            "Terapi laser atau photodynamic therapy (PDT)",
            "Kuretase (pengikisan lesi)",
            "Konsultasi rutin dengan dokter kulit untuk monitoring",
            "Proteksi dari sinar matahari dengan sunscreen SPF 30+",
            "Pemeriksaan kulit secara berkala untuk deteksi dini kanker kulit"
        ]
    },
    "Atopic Dermatitis": {
        "display_name": "Dermatitis Atopik (Eksim)",
        "explanation": "Dermatitis atopik adalah kondisi peradangan kulit kronis yang menyebabkan kulit kering, gatal, dan meradang. Kondisi ini sering terjadi pada anak-anak tetapi dapat berlanjut hingga dewasa. Eksim biasanya muncul di lipatan siku, belakang lutut, leher, dan wajah.",
        "treatment": [
            "Gunakan pelembab secara teratur untuk menjaga kelembaban kulit",
            "Krim kortikosteroid topikal untuk mengurangi peradangan",
            "Antihistamin untuk mengurangi gatal",
            "Hindari pemicu alergi dan iritan (sabun keras, detergen)",
            "Gunakan air hangat (bukan panas) saat mandi",
            "Kompres dingin untuk mengurangi gatal",
            "Konsultasi dengan dokter untuk pengobatan yang lebih intensif jika diperlukan"
        ]
    },
    "Benign keratosis": {
        "display_name": "Keratosis Benigna (Seborrheic Keratosis)",
        "explanation": "Keratosis benigna adalah pertumbuhan kulit non-kanker yang umum terjadi, terutama pada orang dewasa yang lebih tua. Lesi ini muncul sebagai bercak coklat, hitam, atau kuning yang terasa seperti lilin atau kasar saat disentuh. Meskipun tidak berbahaya, beberapa orang memilih untuk menghilangkannya karena alasan kosmetik.",
        "treatment": [
            "Tidak memerlukan pengobatan jika tidak mengganggu",
            "Cryotherapy (pembekuan) untuk menghilangkan lesi",
            "Kuretase (pengikisan) oleh dokter",
            "Terapi laser",
            "Elektrokauter (pembakaran dengan arus listrik)",
            "Konsultasi dengan dokter kulit untuk evaluasi dan pilihan pengobatan",
            "Monitoring jika ada perubahan ukuran, warna, atau bentuk"
        ]
    },
    "Dermatofibroma": {
        "display_name": "Dermatofibroma",
        "explanation": "Dermatofibroma adalah tumor jinak yang umum terjadi pada kulit, biasanya muncul sebagai benjolan kecil, keras, dan berwarna coklat kemerahan. Lesi ini paling sering muncul di kaki dan lengan. Dermatofibroma tidak berbahaya dan biasanya tidak memerlukan pengobatan kecuali jika mengganggu atau berubah.",
        "treatment": [
            "Tidak memerlukan pengobatan jika tidak mengganggu",
            "Eksisi bedah jika mengganggu atau untuk alasan kosmetik",
            "Cryotherapy untuk lesi yang lebih kecil",
            "Monitoring jika ada perubahan ukuran atau warna",
            "Konsultasi dengan dokter kulit untuk evaluasi",
            "Hindari trauma berulang pada area lesi"
        ]
    },
    "Melanocytic nevus": {
        "display_name": "Nevus Melanositik (Tahi Lalat)",
        "explanation": "Nevus melanositik, atau tahi lalat, adalah pertumbuhan kulit yang umum terjadi. Tahi lalat dapat muncul sejak lahir atau berkembang seiring waktu. Sebagian besar tahi lalat adalah jinak, tetapi beberapa dapat berkembang menjadi melanoma. Penting untuk memantau perubahan pada tahi lalat menggunakan metode ABCDE (Asymmetry, Border, Color, Diameter, Evolution).",
        "treatment": [
            "Tidak memerlukan pengobatan jika tidak mengganggu dan tidak berubah",
            "Eksisi bedah jika ada kecurigaan keganasan",
            "Biopsi untuk evaluasi jika ada perubahan",
            "Pemeriksaan kulit rutin oleh dokter (skin check)",
            "Monitoring sendiri dengan metode ABCDE",
            "Fotografi untuk dokumentasi perubahan",
            "Konsultasi segera jika ada perubahan ukuran, warna, atau bentuk"
        ]
    },
    "Melanoma": {
        "display_name": "Melanoma",
        "explanation": "Melanoma adalah jenis kanker kulit yang paling serius, berkembang dari sel melanosit yang menghasilkan pigmen. Melanoma dapat muncul di mana saja di tubuh, termasuk area yang tidak terpapar sinar matahari. Deteksi dini sangat penting karena melanoma dapat menyebar ke bagian tubuh lain jika tidak ditangani.",
        "treatment": [
            "Eksisi bedah untuk mengangkat melanoma dan margin sekitarnya",
            "Biopsi kelenjar getah bening sentinel untuk menentukan staging",
            "Imunoterapi untuk melanoma stadium lanjut",
            "Terapi target (targeted therapy) untuk mutasi gen tertentu",
            "Kemoterapi jika diperlukan",
            "Radioterapi dalam kasus tertentu",
            "Follow-up rutin dan monitoring untuk deteksi kekambuhan",
            "Konsultasi dengan onkologi untuk rencana pengobatan komprehensif"
        ]
    },
    "Squamous cell carcinoma": {
        "display_name": "Karsinoma Sel Skuamosa",
        "explanation": "Karsinoma sel skuamosa (SCC) adalah jenis kanker kulit yang umum, berkembang dari sel skuamosa di lapisan luar kulit. SCC biasanya muncul sebagai bercak merah, bersisik, atau luka yang tidak sembuh. Meskipun dapat menyebar jika tidak ditangani, sebagian besar SCC dapat disembuhkan jika dideteksi dan diobati sejak dini.",
        "treatment": [
            "Eksisi bedah untuk mengangkat kanker dan margin sekitarnya",
            "Mohs surgery untuk kanker di area wajah atau area kritis",
            "Kuretase dan elektrokauter untuk lesi kecil",
            "Cryotherapy untuk lesi superfisial",
            "Radioterapi untuk kasus yang tidak dapat dioperasi",
            "Kemoterapi topikal (5-FU) untuk lesi superfisial",
            "Follow-up rutin untuk monitoring",
            "Proteksi dari sinar matahari untuk pencegahan"
        ]
    },
    "Tinea Ringworm Candidiasis": {
        "display_name": "Tinea / Ringworm / Kandidiasis",
        "explanation": "Tinea adalah infeksi jamur pada kulit yang disebabkan oleh dermatofit. Infeksi ini dapat muncul di berbagai bagian tubuh dan dikenal dengan nama berbeda tergantung lokasinya (tinea corporis, tinea pedis, tinea capitis). Kandidiasis adalah infeksi jamur yang disebabkan oleh Candida, biasanya muncul di area lembab seperti lipatan kulit.",
        "treatment": [
            "Krim antijamur topikal (clotrimazole, miconazole, terbinafine)",
            "Obat antijamur oral untuk infeksi yang lebih parah atau luas",
            "Jaga area tetap bersih dan kering",
            "Gunakan krim sesuai petunjuk, biasanya selama 2-4 minggu",
            "Cuci pakaian, handuk, dan seprai dengan air panas",
            "Hindari berbagi pakaian atau barang pribadi",
            "Konsultasi dengan dokter untuk pengobatan yang tepat",
            "Gunakan sandal di tempat umum yang lembab"
        ]
    },
    "Vascular lesion": {
        "display_name": "Lesi Vaskular",
        "explanation": "Lesi vaskular adalah pertumbuhan atau kelainan yang melibatkan pembuluh darah di kulit. Ini termasuk hemangioma, angioma ceri, spider angioma, dan kondisi vaskular lainnya. Sebagian besar lesi vaskular adalah jinak, tetapi beberapa mungkin memerlukan evaluasi medis.",
        "treatment": [
            "Tidak memerlukan pengobatan jika tidak mengganggu",
            "Terapi laser untuk lesi yang mengganggu secara kosmetik",
            "Sclerotherapy untuk lesi vaskular tertentu",
            "Eksisi bedah untuk lesi yang besar atau mengganggu",
            "Monitoring jika ada perubahan ukuran atau gejala",
            "Konsultasi dengan dokter kulit untuk evaluasi",
            "Hindari trauma pada area lesi"
        ]
    }
}


def get_disease_images(disease_name, num_images=4):
    """
    Get sample images from dataset for a specific disease
    
    Args:
        disease_name: Name of the disease class
        num_images: Number of sample images to return
    
    Returns:
        List of tuples (disease_name, filename)
    """
    disease_path = os.path.join(DATA_DIR, disease_name)
    
    if not os.path.exists(disease_path):
        return []
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(Path(disease_path).glob(ext))
    
    # Randomly select images
    if len(image_files) > num_images:
        selected = random.sample(image_files, num_images)
    else:
        selected = image_files
    
    # Return as (disease_name, filename) tuples
    return [(disease_name, img.name) for img in selected]


@app.route('/api/diseases')
def get_diseases():
    """Get list of all diseases with image counts"""
    diseases = []
    
    for disease_name in class_indices.keys():
        disease_path = os.path.join(DATA_DIR, disease_name)
        image_count = 0
        
        if os.path.exists(disease_path):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_count += len(list(Path(disease_path).glob(ext)))
        
        diseases.append({
            'name': disease_name,
            'display_name': DISEASE_INFO.get(disease_name, {}).get('display_name', disease_name),
            'image_count': image_count
        })
    
    # Sort by name
    diseases.sort(key=lambda x: x['display_name'])
    
    return jsonify({'diseases': diseases})


@app.route('/api/disease/<disease_name>')
def get_disease_info(disease_name):
    """Get information about a specific disease"""
    if disease_name not in DISEASE_INFO:
        return jsonify({'error': 'Disease not found'}), 404
    
    info = DISEASE_INFO[disease_name].copy()
    info['name'] = disease_name
    
    return jsonify(info)


@app.route('/api/disease/<disease_name>/images')
def get_disease_images_api(disease_name):
    """Get sample images for a specific disease"""
    # Decode URL-encoded disease name
    disease_name = disease_name.replace('%20', ' ')
    
    # Ambil list gambar (tuple: disease_name, filename)
    image_paths = get_disease_images(disease_name, num_images=4)
    
    # Convert to URLs (MENGGUNAKAN STATIC URL)
    image_urls = []
    for disease_part, filename in image_paths:
        # Kita buat path relatif terhadap folder static
        # Contoh: dataset/Melanoma/gambar1.jpg
        relative_path = os.path.join('dataset', disease_part, filename)
        
        # Generate URL static yang valid
        # Windows menggunakan backslash (\), kita ganti ke slash (/) untuk URL web
        relative_path = relative_path.replace('\\', '/')
        
        url = url_for('static', filename=relative_path)
        image_urls.append(url)
    
    return jsonify({
        'images': image_urls,
        'total_images': len(image_paths)
    })

# CODE INI SUDAH TIDAK DIPERLUKAN, HAPUS SAJA:
# @app.route('/dataset/<disease_name>/<filename>')
# def serve_image(disease_name, filename):
#     """Serve images from dataset folder"""
#     disease_path = os.path.join(DATA_DIR, disease_name)
#     return send_from_directory(disease_path, filename)


@app.route('/artikel')
def artikel():
    """Artikel edukasi"""
    return render_template('artikel.html')


# ============================================
# Authentication Routes
# ============================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        remember = bool(request.form.get('remember'))
        
        if not username or not password:
            flash('Username dan password harus diisi.', 'error')
            return render_template('login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user, remember=remember)
            next_page = request.args.get('next')
            flash(f'Selamat datang, {user.username}!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Username atau password salah.', 'error')
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page"""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        full_name = request.form.get('full_name', '').strip()
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username harus minimal 3 karakter.')
        if not email or '@' not in email:
            errors.append('Email tidak valid.')
        if not password or len(password) < 6:
            errors.append('Password harus minimal 6 karakter.')
        if password != password_confirm:
            errors.append('Password konfirmasi tidak cocok.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('register.html')
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan.', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar.', 'error')
            return render_template('register.html')
        
        # Create new user
        try:
            user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash('Registrasi berhasil! Silakan login.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('Terjadi kesalahan saat registrasi. Silakan coba lagi.', 'error')
            print(f"Registration error: {e}")
    
    return render_template('register.html')


@app.route('/logout')
@login_required
def logout():
    """Logout"""
    logout_user()
    flash('Anda telah logout.', 'info')
    return redirect(url_for('index'))


# ============================================
# Profile Routes
# ============================================

@app.route('/profile')
@login_required
def profile():
    """Profile page"""
    # Get prediction history (latest 10)
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc())\
        .limit(10)\
        .all()
    
    return render_template('profile.html', user=current_user, predictions=predictions)


@app.route('/profile/edit', methods=['GET', 'POST'])
@login_required
def edit_profile():
    """Edit profile page"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        phone = request.form.get('phone', '').strip()
        current_password = request.form.get('current_password', '')
        new_password = request.form.get('new_password', '')
        new_password_confirm = request.form.get('new_password_confirm', '')
        
        # Update basic info
        if full_name:
            current_user.full_name = full_name
        
        if phone:
            current_user.phone = phone
        
        # Update email if changed
        if email and email != current_user.email:
            if User.query.filter_by(email=email).first():
                flash('Email sudah digunakan oleh user lain.', 'error')
                return redirect(url_for('edit_profile'))
            current_user.email = email
        
        # Update password if provided
        if current_password and new_password:
            if not current_user.check_password(current_password):
                flash('Password lama salah.', 'error')
                return redirect(url_for('edit_profile'))
            
            if len(new_password) < 6:
                flash('Password baru harus minimal 6 karakter.', 'error')
                return redirect(url_for('edit_profile'))
            
            if new_password != new_password_confirm:
                flash('Password konfirmasi tidak cocok.', 'error')
                return redirect(url_for('edit_profile'))
            
            current_user.set_password(new_password)
            flash('Password berhasil diubah.', 'success')
        
        try:
            db.session.commit()
            flash('Profil berhasil diperbarui!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            flash('Terjadi kesalahan saat memperbarui profil.', 'error')
            print(f"Profile update error: {e}")
    
    return render_template('edit_profile.html', user=current_user)


@app.route('/profile/history')
@login_required
def prediction_history():
    """Prediction history page - semua history"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    # Expunge objects from session before parsing to prevent autoflush issues
    for pred in predictions.items:
        db.session.expunge(pred)
        # Parse JSON safely after expunging
        if pred.all_probabilities and isinstance(pred.all_probabilities, str):
            try:
                pred.all_probabilities = json.loads(pred.all_probabilities)
            except:
                pred.all_probabilities = {}
    
    return render_template('prediction_history.html', 
                         predictions=predictions.items,
                         pagination=predictions,
                         user=current_user)


@app.route('/api/profile/history')
@login_required
def get_prediction_history_api():
    """API untuk mendapatkan history prediksi (JSON)"""
    limit = request.args.get('limit', 10, type=int)
    
    predictions = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.created_at.desc())\
        .limit(limit)\
        .all()
    
    return jsonify({
        'success': True,
        'predictions': [pred.to_dict() for pred in predictions]
    })


@app.route('/api/profile/history/<int:history_id>/delete', methods=['POST'])
@login_required
def delete_prediction_history(history_id):
    """Hapus history prediksi"""
    prediction = PredictionHistory.query.get_or_404(history_id)
    
    # Pastikan prediction milik user yang sedang login
    if prediction.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'success': True, 'message': 'History berhasil dihapus'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/diseases/preview')
def get_diseases_preview():
    """Get diseases for preview (all diseases)"""
    diseases = []
    
    for disease_name in class_indices.keys():
        disease_path = os.path.join(DATA_DIR, disease_name)
        image_count = 0
        
        if os.path.exists(disease_path):
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_count += len(list(Path(disease_path).glob(ext)))
        
        info = DISEASE_INFO.get(disease_name, {})
        diseases.append({
            'name': disease_name,
            'display_name': info.get('display_name', disease_name),
            'explanation': info.get('explanation', ''),
            'image_count': image_count
        })
    
    # Sort by name
    diseases.sort(key=lambda x: x['display_name'])
    return jsonify({'diseases': diseases})


# ============================================
# Admin Routes
# ============================================

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard dengan statistik"""
    # Total users
    total_users = User.query.count()
    total_admins = User.query.filter_by(role='admin').count()
    total_regular_users = total_users - total_admins
    
    # Total predictions
    total_predictions = PredictionHistory.query.count()
    
    # Predictions this week
    week_ago = datetime.utcnow() - timedelta(days=7)
    predictions_this_week = PredictionHistory.query.filter(
        PredictionHistory.created_at >= week_ago
    ).count()
    
    # Predictions today
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    predictions_today = PredictionHistory.query.filter(
        PredictionHistory.created_at >= today_start
    ).count()
    
    # Top predicted classes
    top_predictions = db.session.query(
        PredictionHistory.predicted_class,
        func.count(PredictionHistory.id).label('count')
    ).group_by(PredictionHistory.predicted_class)\
     .order_by(desc('count'))\
     .limit(5)\
     .all()
    
    # Recent predictions (latest 10) - eager load user to avoid lazy loading
    recent_predictions = PredictionHistory.query\
        .options(joinedload(PredictionHistory.user))\
        .order_by(PredictionHistory.created_at.desc())\
        .limit(10)\
        .all()
    
    # Users registered this week
    users_this_week = User.query.filter(
        User.created_at >= week_ago
    ).count()
    
    # Expunge objects from session to prevent autoflush when accessing attributes
    # This allows us to safely access user relationship without triggering saves
    for pred in recent_predictions:
        # Access user to ensure it's loaded before expunging
        _ = pred.user
        db.session.expunge(pred)
    
    return render_template('admin/dashboard.html',
                         total_users=total_users,
                         total_admins=total_admins,
                         total_regular_users=total_regular_users,
                         total_predictions=total_predictions,
                         predictions_this_week=predictions_this_week,
                         predictions_today=predictions_today,
                         top_predictions=top_predictions,
                         recent_predictions=recent_predictions,
                         users_this_week=users_this_week)


@app.route('/admin/users')
@admin_required
def admin_users():
    """Admin - User management"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    search = request.args.get('search', '').strip()
    role_filter = request.args.get('role', '')
    
    query = User.query
    
    # Apply filters
    if search:
        query = query.filter(
            (User.username.contains(search)) |
            (User.email.contains(search)) |
            (User.full_name.contains(search))
        )
    
    if role_filter:
        query = query.filter_by(role=role_filter)
    
    users = query.order_by(User.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin/users.html',
                         users=users.items,
                         pagination=users,
                         search=search,
                         role_filter=role_filter)


@app.route('/admin/users/<int:user_id>')
@admin_required
def admin_user_detail(user_id):
    """Admin - Detail user dan prediction history"""
    user = User.query.get_or_404(user_id)
    
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    predictions = PredictionHistory.query.filter_by(user_id=user_id)\
        .order_by(PredictionHistory.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    
    # Expunge objects from session before parsing to prevent autoflush issues
    for pred in predictions.items:
        db.session.expunge(pred)
        # Parse JSON safely after expunging
        if pred.all_probabilities and isinstance(pred.all_probabilities, str):
            try:
                pred.all_probabilities = json.loads(pred.all_probabilities)
            except:
                pred.all_probabilities = {}
    
    # User statistics
    total_predictions = PredictionHistory.query.filter_by(user_id=user_id).count()
    week_ago = datetime.utcnow() - timedelta(days=7)
    predictions_this_week = PredictionHistory.query.filter_by(user_id=user_id)\
        .filter(PredictionHistory.created_at >= week_ago).count()
    
    return render_template('admin/user_detail.html',
                         user=user,
                         predictions=predictions.items,
                         pagination=predictions,
                         total_predictions=total_predictions,
                         predictions_this_week=predictions_this_week)


@app.route('/admin/users/<int:user_id>/toggle-role', methods=['POST'])
@admin_required
def admin_toggle_user_role(user_id):
    """Toggle user role between admin and user"""
    if user_id == current_user.id:
        return jsonify({'error': 'Tidak dapat mengubah role sendiri'}), 400
    
    user = User.query.get_or_404(user_id)
    
    try:
        user.role = 'admin' if user.role == 'user' else 'user'
        db.session.commit()
        return jsonify({
            'success': True,
            'message': f'Role user berhasil diubah menjadi {user.role}',
            'new_role': user.role
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/admin/users/create', methods=['GET', 'POST'])
@admin_required
def admin_create_user():
    """Admin - Create new user"""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'user').strip()
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username harus minimal 3 karakter.')
        if not email or '@' not in email:
            errors.append('Email tidak valid.')
        if not password or len(password) < 6:
            errors.append('Password harus minimal 6 karakter.')
        if password != password_confirm:
            errors.append('Password konfirmasi tidak cocok.')
        if role not in ['user', 'admin']:
            errors.append('Role tidak valid.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('admin/create_user.html', 
                                 username=username, email=email, 
                                 full_name=full_name, phone=phone, role=role)
        
        # Check if username or email already exists
        if User.query.filter_by(username=username).first():
            flash('Username sudah digunakan.', 'error')
            return render_template('admin/create_user.html',
                                 username=username, email=email,
                                 full_name=full_name, phone=phone, role=role)
        
        if User.query.filter_by(email=email).first():
            flash('Email sudah terdaftar.', 'error')
            return render_template('admin/create_user.html',
                                 username=username, email=email,
                                 full_name=full_name, phone=phone, role=role)
        
        # Create new user
        try:
            user = User(
                username=username,
                email=email,
                full_name=full_name if full_name else None,
                phone=phone if phone else None,
                role=role
            )
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            
            flash(f'User "{username}" berhasil dibuat!', 'success')
            return redirect(url_for('admin_users'))
        except Exception as e:
            db.session.rollback()
            flash(f'Terjadi kesalahan saat membuat user: {str(e)}', 'error')
            print(f"Create user error: {e}")
    
    return render_template('admin/create_user.html')


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(user_id):
    """Admin - Edit user"""
    user = User.query.get_or_404(user_id)
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'user').strip()
        password = request.form.get('password', '').strip()
        password_confirm = request.form.get('password_confirm', '').strip()
        
        # Validation
        errors = []
        if not username or len(username) < 3:
            errors.append('Username harus minimal 3 karakter.')
        if not email or '@' not in email:
            errors.append('Email tidak valid.')
        if role not in ['user', 'admin']:
            errors.append('Role tidak valid.')
        if password and len(password) < 6:
            errors.append('Password harus minimal 6 karakter.')
        if password and password != password_confirm:
            errors.append('Password konfirmasi tidak cocok.')
        
        if errors:
            for error in errors:
                flash(error, 'error')
            return render_template('admin/edit_user.html', user=user,
                                 username=username, email=email,
                                 full_name=full_name, phone=phone, role=role)
        
        # Check if username or email already exists (by other user)
        existing_user = User.query.filter_by(username=username).first()
        if existing_user and existing_user.id != user.id:
            flash('Username sudah digunakan oleh user lain.', 'error')
            return render_template('admin/edit_user.html', user=user,
                                 username=user.username, email=email,
                                 full_name=full_name, phone=phone, role=role)
        
        existing_email = User.query.filter_by(email=email).first()
        if existing_email and existing_email.id != user.id:
            flash('Email sudah digunakan oleh user lain.', 'error')
            return render_template('admin/edit_user.html', user=user,
                                 username=username, email=user.email,
                                 full_name=full_name, phone=phone, role=role)
        
        # Update user
        try:
            user.username = username
            user.email = email
            user.full_name = full_name if full_name else None
            user.phone = phone if phone else None
            user.role = role
            
            # Update password if provided
            if password:
                user.set_password(password)
            
            db.session.commit()
            flash(f'User "{username}" berhasil diperbarui!', 'success')
            return redirect(url_for('admin_users'))
        except Exception as e:
            db.session.rollback()
            flash(f'Terjadi kesalahan saat memperbarui user: {str(e)}', 'error')
            print(f"Edit user error: {e}")
    
    return render_template('admin/edit_user.html', user=user)


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    """Delete user"""
    if user_id == current_user.id:
        return jsonify({'error': 'Tidak dapat menghapus akun sendiri'}), 400
    
    user = User.query.get_or_404(user_id)
    
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User berhasil dihapus'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/admin/predictions')
@admin_required
def admin_predictions():
    """Admin - All predictions grouped by user"""
    page = request.args.get('page', 1, type=int)
    per_page = 10  # Number of users per page
    search = request.args.get('search', '').strip()
    class_filter = request.args.get('class', '').strip()
    
    # Build base query for users with predictions
    user_subquery = db.session.query(PredictionHistory.user_id).distinct()
    
    # Apply class filter to subquery
    if class_filter:
        user_subquery = user_subquery.filter(PredictionHistory.predicted_class == class_filter)
    
    # Apply search filter to predictions if searching by prediction class
    if search:
        # Check if search matches any prediction class
        user_subquery = user_subquery.filter(
            PredictionHistory.predicted_class.contains(search)
        )
    
    # Get user IDs
    user_ids = [uid[0] for uid in user_subquery.all()]
    
    # Query users
    user_query = User.query.filter(User.id.in_(user_ids))
    
    # Apply search filter to users
    if search:
        user_query = user_query.filter(
            (User.username.contains(search)) |
            (User.email.contains(search)) |
            (User.full_name.contains(search))
        )
    
    # Get all matching users
    all_users = user_query.order_by(User.username).all()
    
    # If search didn't match users but matched predictions, include those users
    if search and not all_users:
        # Get user IDs from predictions that match search
        pred_user_ids = db.session.query(PredictionHistory.user_id)\
            .filter(PredictionHistory.predicted_class.contains(search))\
            .distinct().all()
        pred_user_ids = [uid[0] for uid in pred_user_ids]
        if pred_user_ids:
            all_users = User.query.filter(User.id.in_(pred_user_ids)).order_by(User.username).all()
    
    # Paginate users
    total_users = len(all_users)
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_users = all_users[start_idx:end_idx]
    
    # Group predictions by user
    users_predictions = {}
    for user in paginated_users:
        # Get predictions for this user
        pred_query = PredictionHistory.query.filter_by(user_id=user.id)
        
        # Apply class filter
        if class_filter:
            pred_query = pred_query.filter(PredictionHistory.predicted_class == class_filter)
        
        # Apply search filter to predictions
        if search:
            # Check if search matches user info
            user_matches = (
                search.lower() in user.username.lower() or
                search.lower() in (user.email or '').lower() or
                search.lower() in (user.full_name or '').lower()
            )
            # If user doesn't match, filter by prediction class
            if not user_matches:
                pred_query = pred_query.filter(
                    PredictionHistory.predicted_class.contains(search)
                )
        
        predictions = pred_query.order_by(PredictionHistory.created_at.desc()).limit(50).all()
        
        # Expunge and parse
        for pred in predictions:
            db.session.expunge(pred)
            if pred.all_probabilities and isinstance(pred.all_probabilities, str):
                try:
                    pred.all_probabilities = json.loads(pred.all_probabilities)
                except:
                    pred.all_probabilities = {}
        
        if predictions:
            users_predictions[user] = predictions
    
    # Calculate pagination info
    total_pages = (total_users + per_page - 1) // per_page if total_users > 0 else 1
    
    # Get all unique classes for filter
    all_classes = db.session.query(PredictionHistory.predicted_class)\
        .distinct()\
        .order_by(PredictionHistory.predicted_class)\
        .all()
    all_classes = [c[0] for c in all_classes]
    
    # Create pagination object-like structure
    class Pagination:
        def __init__(self, page, pages, has_prev, has_next):
            self.page = page
            self.pages = pages
            self.has_prev = has_prev
            self.has_next = has_next
            self.prev_num = page - 1 if has_prev else None
            self.next_num = page + 1 if has_next else None
    
    pagination = Pagination(
        page=page,
        pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages
    )
    
    return render_template('admin/predictions.html',
                         users_predictions=users_predictions,
                         pagination=pagination,
                         search=search,
                         class_filter=class_filter,
                         all_classes=all_classes)


@app.route('/admin/predictions/<int:prediction_id>/delete', methods=['POST'])
@admin_required
def admin_delete_prediction(prediction_id):
    """Delete prediction"""
    prediction = PredictionHistory.query.get_or_404(prediction_id)
    
    try:
        db.session.delete(prediction)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Prediksi berhasil dihapus'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
    
with app.app_context():
    try:
        db.create_all()
        print("✅ Database & Tabel berhasil dibuat!")
    except Exception as e:
        print(f"⚠️ Gagal membuat database: {e}")

if __name__ == '__main__':
    # Create database tables
    with app.app_context():
        db.create_all()
        print("✅ Database tables created/verified")
        
        # Add role column if it doesn't exist (for existing databases)
        try:
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('users')]
            if 'role' not in columns:
                print("⚠️  Adding 'role' column to users table...")
                db.engine.execute('ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT "user"')
                print("✅ Role column added")
        except Exception as e:
            print(f"⚠️  Could not add role column (might already exist): {e}")
    
    print("\n🚀 Starting Flask application...")
    print("📝 Model status:", "✅ Loaded" if model is not None else "❌ Not loaded")
    print("📝 Classes:", len(class_indices), "classes")
    print("\n🌐 Server running at http://127.0.0.1:5000")
    print("📊 Admin panel: http://127.0.0.1:5000/admin (login as admin first)")
    print("💡 To create admin user, run: python create_admin.py")
    app.run(debug=True, host='0.0.0.0', port=5000)

