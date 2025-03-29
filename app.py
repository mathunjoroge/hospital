from flask import Flask, render_template, request, redirect, url_for, flash
from config import Config
import os
from sqlalchemy.orm import Session
from flask_login import login_user, current_user, logout_user, login_required
from extensions import db, login_manager  # Import db and login_manager from extensions
from werkzeug.security import check_password_hash
from departments.models.user import User  # Import User model
from departments.models.admin import Log  # Assuming a Log model exists or will be added
from departments.models.nursing import Notifications
import logging
from flask_mail import Mail
from flask_wtf.csrf import CSRFProtect
from datetime import datetime  # Import datetime module

app = Flask(__name__)
app.config.from_object(Config)
mail = Mail(app)
@app.template_filter('parse_iso')
def parse_iso(timestamp):
    try:
        # Handle timestamps with time zones by replacing 'Z' with '+00:00'
        if isinstance(timestamp, str):
            timestamp = timestamp.replace('Z', '+00:00')
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return timestamp
    except (ValueError, TypeError) as e:
        print(f"Error parsing timestamp {timestamp}: {e}")
        return timestamp



# Create the Flask app
app = Flask(__name__)
app.config.from_object(Config)
@app.context_processor
def inject_unread_notifications():
    try:
        if current_user.is_authenticated and current_user.role == 'nursing':
            unread_notifications = Notifications.query.filter_by(receiver_id=current_user.id, is_read=False).count()
        else:
            unread_notifications = 0
        return dict(unread_notifications=unread_notifications)
    except Exception as e:
        print(f"Error in inject_unread_notifications: {str(e)}")
        return dict(unread_notifications=0)
# Define the datetime filter
def datetime_filter(value):
    if value == "now":
        return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')  # Customize format as needed
    return value

# Register the filter with Jinja2
app.jinja_env.filters['datetime'] = datetime_filter    
# Define the upload folder path
app.config['UPLOAD_FOLDER'] = os.path.join('uploads')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
# Define the upload folder for DICOM files
app.config['DICOM_UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'dicom_uploads')
if not os.path.exists(app.config['DICOM_UPLOAD_FOLDER']):
    os.makedirs(app.config['DICOM_UPLOAD_FOLDER'])  # Create the folder if it doesn't exist

# Set maximum file size (optional, corrected duplicate)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # Limit file size to 32 MB

# Initialize the database and login manager with the app
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the user loader function
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Define the home route
@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for(f'{current_user.role}.index'))
    else:
        return redirect(url_for('login'))

# Define the login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        try:
            # Query the user by username
            user = User.query.filter_by(username=username).first()

            # Verify the password using check_password_hash
            if user and check_password_hash(user.password, password):
                # Log the user in
                login_user(user)
                # Log successful login to database and console
                db.session.add(Log(
                    level='INFO',
                    message=f"User {user.username} (ID: {user.id}) logged in successfully",
                    user_id=user.id,
                    source='auth'
                ))
                db.session.commit()
                logger.info(f"User {user.id} ({user.username}) logged in successfully")
                return redirect(url_for(f'{user.role}.index'))
            else:
                # Log failed attempt
                db.session.add(Log(
                    level='WARNING',
                    message=f"Failed login attempt for username: {username}",
                    user_id=None,
                    source='auth'
                ))
                db.session.commit()
                logger.warning(f"Failed login attempt for username: {username}")
                flash('Invalid credentials', 'error')

        except Exception as e:
            db.session.rollback()
            db.session.add(Log(
                level='ERROR',
                message=f"Login error: {str(e)}",
                user_id=None,
                source='auth'
            ))
            db.session.commit()
            logger.error(f"Error in login: {str(e)}", exc_info=True)
            flash(f'Login error: {e}', 'error')

    return render_template('login.html')

# Define the logout route
@app.route('/logout')
@login_required
def logout():
    try:
        user_id = current_user.id
        username = current_user.username
        logout_user()  # Log the user out
        # Log logout event
        db.session.add(Log(
            level='INFO',
            message=f"User {username} (ID: {user_id}) logged out",
            user_id=user_id,
            source='auth'
        ))
        db.session.commit()
        logger.info(f"User {user_id} ({username}) logged out")
        return redirect(url_for('login'))
    except Exception as e:
        db.session.rollback()
        db.session.add(Log(
            level='ERROR',
            message=f"Logout error: {str(e)}",
            user_id=None,
            source='auth'
        ))
        db.session.commit()
        logger.error(f"Error in logout: {str(e)}", exc_info=True)
        flash(f'Logout error: {e}', 'error')
        return redirect(url_for('login'))

# Import and register blueprints
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
#register blueprints
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create all tables
    app.run(debug=True)

