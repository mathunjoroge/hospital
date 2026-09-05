from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_wtf.csrf import CSRFProtect

# Database instance
db = SQLAlchemy()

# Login manager instance
login_manager = LoginManager()

# SocketIO instance
socketio = SocketIO()

# CSRF Protection instance
csrf = CSRFProtect()
