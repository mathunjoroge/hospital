from extensions import db 
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# Imaging Results Table
class ImagingResult(db.Model):
    __tablename__ = 'imaging_results'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.String, unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    imaging_id = db.Column(db.Integer, db.ForeignKey('imaging.id'), nullable=False)
    test_date = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    result_notes = db.Column(db.Text)
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

    # Relationships
    patient = db.relationship('Patient', backref='imaging_results')
    imaging = db.relationship('Imaging', backref='imaging_results')
    updated_by_user = db.relationship('User', backref='imaging_results_updated')
    requested_image = db.relationship('RequestedImage', backref='imaging_result', uselist=False)

    def __repr__(self):
        return f"<ImagingResult {self.result_id} - Patient {self.patient_id}, Imaging {self.imaging_id}>"