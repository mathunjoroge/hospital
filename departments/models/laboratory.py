from extensions import db 
from datetime import datetime
from sqlalchemy.orm import relationship

class LabResultTemplate(db.Model):
    __tablename__ = 'labresults_templates'
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey('labtests.id'), nullable=False)
    parameter_name = db.Column(db.String(255), nullable=False)
    normal_range_low = db.Column(db.Float, nullable=False)
    normal_range_high = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(50), nullable=False)
    # Relationship to LabTest
    lab_test = db.relationship('LabTest', backref='result_templates')

    def __repr__(self):
        return f"<LabResultTemplate {self.parameter_name} for Test {self.test_id}>"
    
class LabResult(db.Model):
    __tablename__ = 'lab_results'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    lab_test_id = db.Column(db.Integer, db.ForeignKey('labtests.id'), nullable=False)
    test_date = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    result_notes = db.Column(db.Text)  # Optional notes

    # Newly added columns
    result_id = db.Column(db.String, unique=True, nullable=False)  # Unique identifier for result
    result = db.Column(db.Text, nullable=True)  # Stores the test result
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Who updated the result

    # Relationships (if needed)
    patient = db.relationship('Patient', backref=db.backref('lab_results', lazy=True))
    lab_test = db.relationship('LabTest', backref=db.backref('lab_results', lazy=True))
    updated_by_user = db.relationship('User', backref=db.backref('updated_results', lazy=True))

    def __repr__(self):
        return f"<LabResult {self.id} - Patient {self.patient_id}, Test {self.lab_test_id}>"  

