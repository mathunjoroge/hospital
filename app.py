from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import login_user, current_user, logout_user, login_required
from flask_migrate import Migrate
from flask_mail import Mail
from flask_socketio import SocketIO
from werkzeug.security import check_password_hash
from datetime import datetime
import os
import logging
import uuid
from config import Config
from extensions import db, login_manager
from departments.models.user import User
from departments.models.admin import Log
from departments.models.nursing import Notifications
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Upload folders
app.config['UPLOAD_FOLDER'] = os.path.join('uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['DICOM_UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'dicom_uploads')
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB file limit

os.makedirs(app.config['DICOM_UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
mail = Mail(app)
migrate = Migrate(app, db)
socketio = SocketIO(app)  # SocketIO initialized with app

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Jinja filters
@app.template_filter('parse_iso')
def parse_iso(timestamp):
    try:
        if isinstance(timestamp, str):
            timestamp = timestamp.replace('Z', '+00:00')
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        return timestamp
    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return timestamp

@app.context_processor
def inject_unread_notifications():
    try:
        if current_user.is_authenticated and current_user.role == 'nursing':
            count = Notifications.query.filter_by(receiver_id=current_user.id, is_read=False).count()
        else:
            count = 0
        return dict(unread_notifications=count)
    except Exception as e:
        print(f"Error in inject_unread_notifications: {e}")
        return dict(unread_notifications=0)

def datetime_filter(value):
    if value == "now":
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return value

app.jinja_env.filters['datetime'] = datetime_filter

# Login user loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for(f'{current_user.role}.index'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        try:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                login_user(user)
                db.session.add(Log(level='INFO', message=f"User {user.username} (ID: {user.id}) logged in", user_id=user.id, source='auth'))
                db.session.commit()
                logger.info(f"User {user.id} ({user.username}) logged in")
                return redirect(url_for(f'{user.role}.index'))
            else:
                db.session.add(Log(level='WARNING', message=f"Failed login attempt: {username}", source='auth'))
                db.session.commit()
                logger.warning(f"Failed login attempt for {username}")
                flash('Invalid credentials', 'error')
        except Exception as e:
            db.session.rollback()
            db.session.add(Log(level='ERROR', message=f"Login error: {e}", source='auth'))
            db.session.commit()
            logger.error(f"Login error: {e}", exc_info=True)
            flash(f'Login error: {e}', 'error')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    try:
        user_id = current_user.id
        username = current_user.username
        logout_user()
        db.session.add(Log(level='INFO', message=f"User {username} (ID: {user_id}) logged out", user_id=user_id, source='auth'))
        db.session.commit()
        logger.info(f"User {user_id} ({username}) logged out")
        return redirect(url_for('login'))
    except Exception as e:
        db.session.rollback()
        db.session.add(Log(level='ERROR', message=f"Logout error: {e}", source='auth'))
        db.session.commit()
        logger.error(f"Logout error: {e}", exc_info=True)
        flash(f'Logout error: {e}', 'error')
        return redirect(url_for('login'))

if __name__ == '__main__':
    # Import blueprints only when running the app
    from departments.records import bp as records_bp
    from departments.billing import bp as billing_bp
    from departments.pharmacy import bp as pharmacy_bp
    from departments.medicine import bp as medicine_bp
    from departments.laboratory import bp as laboratory_bp
    from departments.imaging import bp as imaging_bp
    from departments.stores import bp as stores_bp
    from departments.admin import bp as admin_bp
    from departments.nursing import bp as nursing_bp
    from departments.hr import bp as hr_bp
    from departments.api import bp as api_bp

    # Register blueprints after all imports and definitions
    app.register_blueprint(records_bp, url_prefix='/records')
    app.register_blueprint(billing_bp, url_prefix='/billing')
    app.register_blueprint(pharmacy_bp, url_prefix='/pharmacy')
    app.register_blueprint(medicine_bp, url_prefix='/medicine')
    app.register_blueprint(laboratory_bp, url_prefix='/laboratory')
    app.register_blueprint(imaging_bp, url_prefix='/imaging')
    app.register_blueprint(stores_bp, url_prefix='/stores')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(nursing_bp, url_prefix='/nursing')
    app.register_blueprint(hr_bp, url_prefix='/hr')
    app.register_blueprint(api_bp, url_prefix='/api')

    socketio.run(app, debug=True)  # Use socketio.run