import os

# Define the base directory for the project
BASE_DIR = "hims"

# List of departments
DEPARTMENTS = [
    "records",
    "nursing",
    "pharmacy",
    "medicine",
    "laboratory",
    "imaging",
    "mortuary",
    "hr",
    "stores"
]

# Function to create a file with content
def create_file(file_path, content=""):
    with open(file_path, "w") as f:
        f.write(content)

# Function to create department structure
def create_department_structure(department_name):
    department_dir = os.path.join(BASE_DIR, "departments", department_name)
    
    # Create department folder
    os.makedirs(os.path.join(department_dir, "templates"), exist_ok=True)
    
    # Create __init__.py
    init_content = f"""
from flask import Blueprint

bp = Blueprint('{department_name}', __name__, template_folder='templates')

from . import routes
"""
    create_file(os.path.join(department_dir, "__init__.py"), init_content.strip())
    
    # Create models.py
    models_content = """
from ..models import db

class DepartmentData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(db.Text, nullable=False)
"""
    create_file(os.path.join(department_dir, "models.py"), models_content.strip())
    
    # Create routes.py
    routes_content = f"""
from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from ..models import DepartmentData
from . import bp

@bp.route('/')
@login_required
def index():
    if current_user.role != '{department_name}':
        return redirect(url_for('login'))
    return render_template('{department_name}.html')
"""
    create_file(os.path.join(department_dir, "routes.py"), routes_content.strip())
    
    # Create templates/<department_name>.html
    template_content = """
{% extends "../../base.html" %}
{% block content %}
<h2>{{ '{{ department_name|title }}' }} Department</h2>
<p>Welcome to the {{ '{{ department_name|title }}' }} Department.</p>
{% endblock %}
"""
    create_file(os.path.join(department_dir, "templates", f"{department_name}.html"), template_content.strip())

# Main function to create the entire project structure
def create_project_structure():
    print("Creating project structure...")
    
    # Create base directories
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "departments"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "ml_models"), exist_ok=True)
    
    # Create app.py
    app_content = """
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Import and register blueprints
"""
    for department in DEPARTMENTS:
        app_content += f"from departments.{department} import bp as {department}_bp\n"
    app_content += "\n"

    for department in DEPARTMENTS:
        app_content += f"app.register_blueprint({department}_bp, url_prefix='/{department}')\n"
    
    app_content += """
@login_manager.user_loader
def load_user(user_id):
    from departments.models import User
    return User.query.get(int(user_id))

from departments.models import db

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
"""
    create_file(os.path.join(BASE_DIR, "app.py"), app_content.strip())
    
    # Create config.py
    config_content = """
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///hims.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
"""
    create_file(os.path.join(BASE_DIR, "config.py"), config_content.strip())
    
    # Create requirements.txt
    requirements_content = """
Flask==2.3.2
SQLAlchemy==1.4.46
Flask-Login==0.6.2
Flask-WTF==1.1.1
pandas==1.5.3
scikit-learn==1.2.2
numpy==1.24.3
"""
    create_file(os.path.join(BASE_DIR, "requirements.txt"), requirements_content.strip())
    
    # Create base.html
    base_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HIMS</title>
</head>
<body>
    <header>
        <h1>Health Information Management System</h1>
        <nav>
            <a href="{{ url_for('logout') }}">Logout</a>
        </nav>
    </header>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
"""
    create_file(os.path.join(BASE_DIR, "templates", "base.html"), base_content.strip())
    
    # Create login.html
    login_content = """
{% extends "base.html" %}
{% block content %}
<h2>Login</h2>
<form method="POST">
    <label>Username:</label>
    <input type="text" name="username" required>
    <br>
    <label>Password:</label>
    <input type="password" name="password" required>
    <br>
    <button type="submit">Login</button>
</form>
{% endblock %}
"""
    create_file(os.path.join(BASE_DIR, "templates", "login.html"), login_content.strip())
    
    # Create departments/models.py
    models_content = """
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # e.g., 'records', 'nursing', etc.
"""
    create_file(os.path.join(BASE_DIR, "departments", "models.py"), models_content.strip())
    
    # Create department-specific structures
    for department in DEPARTMENTS:
        create_department_structure(department)
    
    print("Project structure created successfully!")

# Run the script
if __name__ == "__main__":
    create_project_structure()