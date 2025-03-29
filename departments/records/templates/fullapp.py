from flask import Flask, render_template, request, redirect, url_for, flash
from config import Config
from flask_login import login_user, current_user
from extensions import db, login_manager  # Import db and login_manager from extensions
from werkzeug.security import check_password_hash
from departments.models.user import User  # Import User model

# Create the Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize the database and login manager with the app
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the user loader function
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Import and register blueprints
from departments.records import bp as records_bp
from departments.nursing import bp as nursing_bp
from departments.pharmacy import bp as pharmacy_bp
from departments.medicine import bp as medicine_bp
from departments.laboratory import bp as laboratory_bp
from departments.imaging import bp as imaging_bp
from departments.mortuary import bp as mortuary_bp
from departments.hr import bp as hr_bp
from departments.stores import bp as stores_bp

app.register_blueprint(records_bp, url_prefix='/records')
app.register_blueprint(nursing_bp, url_prefix='/nursing')
app.register_blueprint(pharmacy_bp, url_prefix='/pharmacy')
app.register_blueprint(medicine_bp, url_prefix='/medicine')
app.register_blueprint(laboratory_bp, url_prefix='/laboratory')
app.register_blueprint(imaging_bp, url_prefix='/imaging')
app.register_blueprint(mortuary_bp, url_prefix='/mortury')
app.register_blueprint(hr_bp, url_prefix='/hr')
app.register_blueprint(stores_bp, url_prefix='/stores')

# Define the homepage route to render the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Query the user by username
        user = User.query.filter_by(username=username).first()

        # Verify the password using check_password_hash
        if user and check_password_hash(user.password, password):
            # Log the user in
            login_user(user)

            # Redirect based on role (corrected endpoint)
            return redirect(url_for(f'{user.role}.index'))  # Use 'records.index', 'nursing.index', etc.
        else:
            # Flash an error message for invalid credentials
            flash('Invalid credentials', 'error')

    # Render the login page for GET requests or failed logins
    return render_template('login.html')

# Define the logout route
@app.route('/logout')
def logout():
    from flask_login import logout_user
    logout_user()  # Log the user out
    return redirect(url_for('login'))  # Redirect to the login page after logout

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create all tables
    app.run(debug=True)