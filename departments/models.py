from flask_login import UserMixin
from extensions import db  # Import db from extensions

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(50), nullable=False)  # e.g., 'records', 'nursing', etc.

# Add other models here...
# Add other models here...
# Records Department Model
class Record(db.Model):
    __tablename__ = 'records'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    data = db.Column(db.Text, nullable=False)

# Nursing Department Model
class NursingData(db.Model):
    __tablename__ = 'nursing_data'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    notes = db.Column(db.Text, nullable=False)

# Pharmacy Department Model
class PharmacyData(db.Model):
    __tablename__ = 'pharmacy_data'
    id = db.Column(db.Integer, primary_key=True)
    medication = db.Column(db.String(100), nullable=False)
    stock = db.Column(db.Integer, nullable=False)

# Medicine (Doctors) Department Model
class DiagnosisData(db.Model):
    __tablename__ = 'diagnosis_data'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    diagnosis = db.Column(db.Text, nullable=False)

# Laboratory Department Model
class LabResult(db.Model):
    __tablename__ = 'lab_results'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    test_name = db.Column(db.String(100), nullable=False)
    result = db.Column(db.Text, nullable=False)

# Imaging Department Model
class ImagingData(db.Model):
    __tablename__ = 'imaging_data'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    scan_type = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(200), nullable=False)

# Mortuary Department Model
class MortuaryData(db.Model):
    __tablename__ = 'mortuary_data'
    id = db.Column(db.Integer, primary_key=True)
    deceased_id = db.Column(db.String(50), nullable=False)
    date_of_death = db.Column(db.Date, nullable=False)
    cause_of_death = db.Column(db.Text, nullable=False)

# Human Resource Department Model
class EmployeeData(db.Model):
    __tablename__ = 'employee_data'
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    position = db.Column(db.String(100), nullable=False)

# Stores Department Model
class InventoryData(db.Model):
    __tablename__ = 'inventory_data'
    id = db.Column(db.Integer, primary_key=True)
    item_name = db.Column(db.String(100), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    location = db.Column(db.String(100), nullable=False)