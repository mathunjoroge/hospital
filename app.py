import logging
import os
import shutil
from datetime import datetime, timedelta, timezone

import dotenv
import pyotp
import redis
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_apscheduler import APScheduler
from flask_login import current_user, login_required, login_user, logout_user
from flask_mail import Mail
from flask_migrate import Migrate
from flask_session import Session
from markupsafe import Markup, escape
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import check_password_hash

from config import Config
from departments.models.admin import Log
from departments.models.nursing import Notifications
from departments.models.user import User
from extensions import csrf, db, jwt, limiter, login_manager, socketio

dotenv.load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Apply ProxyFix for correct rate-limiting and logging behind reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Upload folders
app.config["UPLOAD_FOLDER"] = os.path.join("Uploads")
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}
app.config["DICOM_UPLOAD_FOLDER"] = os.path.join(
    app.root_path, "static", "dicom_Uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB file limit

secret_key = os.environ.get("SECRET_KEY")
if not secret_key:
    if os.environ.get("FLASK_ENV") == "testing":
        secret_key = "test-secret-key-not-for-production"
    else:
        raise RuntimeError(
            "SECRET_KEY environment variable must be set "
            "(FLASK_ENV=testing is the only exception)."
        )

app.config["SECRET_KEY"] = secret_key
app.config["ENABLE_TELEMEDICINE"] = (
    os.environ.get(
        "ENABLE_TELEMEDICINE", os.environ.get("TELEMEDICINE_ENABLED", "false")
    ).lower()
    == "true"
)
app.config["SESSION_TYPE"] = "redis"
redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))
app.config["SESSION_REDIS"] = redis.Redis(host=redis_host, port=redis_port, db=0)
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = 1800  # 30 minutes
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.environ.get("FLASK_ENV", "development") == "production"
)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_USE_SIGNER"] = True
app.config["SESSION_FILE_THRESHOLD"] = 500
app.config["SESSION_KEY_PREFIX"] = "hospital_flask_session:"

# Temporary logger before full logging config
temp_logger = logging.getLogger(__name__)

filesystem_session_dir = os.path.abspath(os.path.join(os.getcwd(), "flask_session"))

try:
    app.config["SESSION_REDIS"].ping()
    temp_logger.info("Redis connection successful. Using Redis session backend.")
    os.makedirs(filesystem_session_dir, exist_ok=True)
    if not os.access(filesystem_session_dir, os.W_OK):
        temp_logger.warning(
            f"Filesystem session directory not writable: {filesystem_session_dir}"
        )
    else:
        temp_logger.debug(
            f"Filesystem session directory ready at: {filesystem_session_dir}"
        )
except redis.ConnectionError as e:
    temp_logger.error(
        f"Redis connection failed: {e}. Falling back to filesystem sessions."
    )
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = filesystem_session_dir
    os.makedirs(filesystem_session_dir, exist_ok=True)
    if not os.access(filesystem_session_dir, os.W_OK):
        temp_logger.critical(
            f"Session directory {filesystem_session_dir} is NOT writable."
        )
        raise PermissionError(
            f"Session directory {filesystem_session_dir} is not writable"
        )
    else:
        temp_logger.info(f"Using Filesystem sessions at: {filesystem_session_dir}")

Session(app)


# Ensure DICOM upload folder exists
os.makedirs(app.config["DICOM_UPLOAD_FOLDER"], exist_ok=True)


# Initialize extensions
db.init_app(app)
from departments.audit import register_audit_listeners  # noqa: E402

register_audit_listeners()
csrf.init_app(app)
# Exempt the JWT token endpoint from CSRF — API clients don't carry CSRF cookies
from departments.api.auth import get_token as _api_get_token  # noqa: E402

csrf.exempt(_api_get_token)

limiter.init_app(app)
if app.config.get("TESTING"):
    limiter.enabled = False

login_manager.init_app(app)
login_manager.login_view = "login"

# JWT configuration — isolated from session SECRET_KEY for security
# Falls back to SECRET_KEY if not explicitly set in environment
jwt_secret = os.environ.get("JWT_SECRET_KEY", secret_key)
app.config["JWT_SECRET_KEY"] = jwt_secret
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=24)
app.config["JWT_TOKEN_LOCATION"] = ["headers"]
app.config["JWT_HEADER_NAME"] = "Authorization"
app.config["JWT_HEADER_TYPE"] = "Bearer"
jwt.init_app(app)
mail = Mail(app)
migrate = Migrate(app, db)
socketio.init_app(app)

# Initialize APScheduler
scheduler = APScheduler()
scheduler.init_app(app)


@scheduler.task("cron", id="check_staff_credentials_daily", hour=1, minute=0)
def scheduled_staff_credential_check():
    with app.app_context():
        from departments.notifications.triggers import (
            trigger_staff_credential_expiry_check,
        )

        trigger_staff_credential_expiry_check()


if not scheduler.running and not app.config.get("TESTING"):
    try:
        scheduler.start()
    except Exception:
        pass

# Logging (Moved here for proper config application)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("medical_chatbot.log"), logging.StreamHandler()],
)
# Re-get the logger now that basicConfig is set
logger = logging.getLogger(__name__)
logger.debug(
    f"Final Session configuration: TYPE={app.config['SESSION_TYPE']}, REDIS={app.config.get('SESSION_REDIS', 'Filesystem')}"
)


# Jinja filters
@app.template_filter("nl2br_safe")
def nl2br_safe(text):
    if text is None:
        return ""
    return Markup("<br>".join(escape(str(text)).split("\n")))  # nosec B704


@app.template_filter("parse_iso")
def parse_iso(timestamp):
    try:
        if isinstance(timestamp, str):
            timestamp = timestamp.replace("Z", "+00:00")
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return timestamp
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing timestamp {timestamp}: {e}")
        return timestamp


@app.context_processor
def inject_unread_notifications():
    try:
        if current_user.is_authenticated and current_user.role == "nursing":
            count = Notifications.query.filter_by(
                receiver_id=current_user.id, is_read=False
            ).count()
        else:
            count = 0
        return dict(unread_notifications=count)
    except Exception as e:
        logger.error(f"Error in inject_unread_notifications: {e}")
        return dict(unread_notifications=0)


def datetime_filter(value):
    if value == "now":
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return value


app.jinja_env.filters["datetime"] = datetime_filter


# Login user loader
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# Routes
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for(f"{current_user.role}.index"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        try:
            user = User.query.filter_by(username=username).first()
            if user:
                if user.locked_until and user.locked_until.replace(
                    tzinfo=timezone.utc
                ) > datetime.now(timezone.utc):
                    flash(
                        "Account locked due to too many failed attempts. Please try again later.",
                        "error",
                    )
                    return render_template("login.html")

                if check_password_hash(user.password, password):
                    user.failed_login_attempts = 0
                    user.locked_until = None
                    db.session.commit()

                    if user.mfa_enabled or (user.role == "admin" and user.totp_secret):
                        session["mfa_pending_user_id"] = user.id
                        return redirect(url_for("mfa_verify"))

                    login_user(user)
                    db.session.add(
                        Log(
                            level="INFO",
                            message=f"User {user.username} (ID: {user.id}) logged in",
                            user_id=user.id,
                            source="auth",
                        )
                    )
                    db.session.commit()
                    logger.info(f"User {user.id} ({user.username}) logged in")
                    return redirect(url_for(f"{user.role}.index"))
                else:
                    user.failed_login_attempts += 1
                    if user.failed_login_attempts >= 5:
                        user.locked_until = datetime.now(timezone.utc) + timedelta(
                            minutes=15
                        )
                    db.session.commit()

            db.session.add(
                Log(
                    level="WARNING",
                    message=f"Failed login attempt: {username}",
                    source="auth",
                )
            )
            db.session.commit()
            logger.warning(f"Failed login attempt for {username}")
            flash("Invalid credentials", "error")
        except Exception as e:
            db.session.rollback()
            try:
                db.session.add(
                    Log(level="ERROR", message=f"Login error: {e}", source="auth")
                )
                db.session.commit()
            except Exception:
                db.session.rollback()
            logger.error(f"Login error: {e}", exc_info=True)
            flash("Something went wrong. Please try again.", "error")
    return render_template("login.html")


@app.route("/mfa_verify", methods=["GET", "POST"])
def mfa_verify():
    user_id = session.get("mfa_pending_user_id")
    if not user_id:
        return redirect(url_for("login"))

    user = db.session.get(User, user_id)
    if not user or not user.totp_secret:
        session.pop("mfa_pending_user_id", None)
        return redirect(url_for("login"))

    if request.method == "POST":
        code = request.form.get("code", "").strip()
        totp = pyotp.TOTP(user.totp_secret)
        if totp.verify(code):
            session.pop("mfa_pending_user_id", None)
            login_user(user)
            db.session.add(
                Log(
                    level="INFO",
                    message=f"User {user.username} (ID: {user.id}) logged in with MFA",
                    user_id=user.id,
                    source="auth",
                )
            )
            db.session.commit()
            logger.info(f"User {user.id} ({user.username}) logged in with MFA")
            return redirect(url_for(f"{user.role}.index"))
        else:
            flash("Invalid MFA verification code. Please try again.", "error")

    return render_template("mfa_verify.html", username=user.username)


@app.route("/logout")
@login_required
def logout():
    try:
        user_id = current_user.id
        username = current_user.username
        logout_user()
        db.session.add(
            Log(
                level="INFO",
                message=f"User {username} (ID: {user_id}) logged out",
                user_id=user_id,
                source="auth",
            )
        )
        db.session.commit()
        logger.info(f"User {user_id} ({username}) logged out")
        return redirect(url_for("login"))
    except Exception as e:
        db.session.rollback()
        db.session.add(Log(level="ERROR", message=f"Logout error: {e}", source="auth"))
        db.session.commit()
        logger.error(f"Logout error: {e}", exc_info=True)
        flash("Something went wrong. Please try again.", "error")


@app.route("/healthz", methods=["GET"])
def healthz():
    """
    Production Health Check Endpoint.
    Verifies DB connectivity, disk space availability, and system status.
    """
    db_status = "connected"
    http_code = 200

    # 1. Test database ping query
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_status = f"disconnected: {e}"
        http_code = 503

    # 2. Check disk space
    try:
        total, used, free = shutil.disk_usage(".")
        total_gb = round(total / (1024**3), 2)
        free_gb = round(free / (1024**3), 2)
        percent_free = round((free / total) * 100, 1)
    except Exception:
        total_gb, free_gb, percent_free = 0, 0, 0

    status_str = "ok" if http_code == 200 else "degraded"

    return jsonify(
        {
            "status": status_str,
            "system": "HMIS",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "database": db_status,
                "disk": {
                    "total_gb": total_gb,
                    "free_gb": free_gb,
                    "percent_free": percent_free,
                },
            },
        }
    ), http_code


from departments.admin import bp as admin_bp  # noqa: E402
from departments.analytics import bp as analytics_bp  # noqa: E402
from departments.api import bp as api_bp  # noqa: E402
from departments.api.dhis2_exporter import khis_bp  # noqa: E402
from departments.api.fhir import fhir_bp  # noqa: E402
from departments.billing import bp as billing_bp  # noqa: E402
from departments.billing.mpesa import mpesa_bp  # noqa: E402
from departments.emergency import bp as emergency_bp  # noqa: E402
from departments.hr import bp as hr_bp  # noqa: E402
from departments.imaging import bp as imaging_bp  # noqa: E402
from departments.imaging.dicom import dicom_bp  # noqa: E402
from departments.laboratory import bp as laboratory_bp  # noqa: E402
from departments.laboratory.panic_alerts import lis_bp  # noqa: E402
from departments.medicine import bp as medicine_bp  # noqa: E402
from departments.medicine.prescribe import prescribe_bp  # noqa: E402
from departments.mortuary import bp as mortuary_bp  # noqa: E402
from departments.nursing import bp as nursing_bp  # noqa: E402
from departments.nursing.mar import mar_bp  # noqa: E402
from departments.nursing.triage import triage_bp  # noqa: E402
from departments.patient_portal import patient_portal_bp  # noqa: E402
from departments.pharmacy import bp as pharmacy_bp  # noqa: E402
from departments.pharmacy.fefo import fefo_bp  # noqa: E402
from departments.pharmacy.po_routes import po_bp  # noqa: E402
from departments.records import bp as records_bp  # noqa: E402
from departments.stores import bp as stores_bp  # noqa: E402
from departments.stores.transfer_routes import transfer_bp  # noqa: E402
from departments.telemedicine import bp as telemedicine_bp  # noqa: E402

app.register_blueprint(records_bp, url_prefix="/records")
app.register_blueprint(billing_bp, url_prefix="/billing")
app.register_blueprint(pharmacy_bp, url_prefix="/pharmacy")
app.register_blueprint(medicine_bp, url_prefix="/medicine")
app.register_blueprint(laboratory_bp, url_prefix="/laboratory")
app.register_blueprint(imaging_bp, url_prefix="/imaging")
app.register_blueprint(stores_bp, url_prefix="/stores")
app.register_blueprint(transfer_bp)  # Phase E: mounts /stores/transfers/*
app.register_blueprint(admin_bp, url_prefix="/admin")

app.register_blueprint(nursing_bp, url_prefix="/nursing")
app.register_blueprint(hr_bp, url_prefix="/hr")
app.register_blueprint(mortuary_bp, url_prefix="/mortuary")
app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(analytics_bp)
app.register_blueprint(patient_portal_bp, url_prefix="/portal")
app.register_blueprint(telemedicine_bp, url_prefix="/telemedicine")  # Phase E
app.register_blueprint(po_bp)  # Phase F: mounts /pharmacy/po/* & /pharmacy/suppliers
app.register_blueprint(
    emergency_bp
)  # Phase D: mounts /emergency/* and /admin/break-glass
app.register_blueprint(mpesa_bp)
app.register_blueprint(triage_bp)
app.register_blueprint(prescribe_bp)
app.register_blueprint(fefo_bp)
app.register_blueprint(lis_bp)
app.register_blueprint(dicom_bp)
app.register_blueprint(mar_bp)
app.register_blueprint(fhir_bp, url_prefix="/api/fhir/R4")
app.register_blueprint(khis_bp, url_prefix="/api/khis")

if __name__ == "__main__":
    with app.app_context():
        try:
            # Schema is managed exclusively by Flask-Migrate (Alembic).
            # Run `flask db upgrade` before starting the app to apply pending migrations.
            from werkzeug.security import generate_password_hash

            from departments.models.user import User

            if not User.query.filter_by(username="admin").first():
                admin_pass = os.environ.get(
                    "DEFAULT_ADMIN_PASSWORD", "AdminPassword123!"
                )
                admin = User(
                    username="admin",
                    password=generate_password_hash(admin_pass, method="pbkdf2:sha256"),
                    role="admin",
                )
                db.session.add(admin)
                db.session.commit()
                print(f"✅ Default admin user created (admin / {admin_pass})")
                print("   Run `flask db upgrade` to ensure the schema is up to date.")
            else:
                print("✅ Database connection verified & admin user exists.")
        except Exception as exc:
            print(f"⚠️  Startup note: {exc}")
            print("   If the database is not initialised, run: flask db upgrade")

    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    socketio.run(app, debug=debug_mode)
