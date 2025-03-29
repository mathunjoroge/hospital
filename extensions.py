from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Database instance
db = SQLAlchemy()

# Login manager instance
login_manager = LoginManager()