from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Database instance
db = SQLAlchemy()

# Login manager instance
login_manager = LoginManager()

# SocketIO instance
socketio = SocketIO()

# CSRF Protection instance
csrf = CSRFProtect()

# Limiter instance
limiter = Limiter(key_func=get_remote_address)

