from datetime import datetime

from extensions import db


class MortuaryData(db.Model):
    __tablename__ = 'mortuary_data'
    id = db.Column(db.Integer, primary_key=True)
    deceased_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False, index=True)
    date_of_death = db.Column(db.Date, nullable=False)
    cause_of_death = db.Column(db.Text, nullable=False)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Staff who recorded

    # Relationships
    patient = db.relationship('Patient', backref=db.backref('mortuary_record', uselist=False))
    recorded_by_user = db.relationship('User', backref='mortuary_entries')

    def __repr__(self):
        return f'<MortuaryData {self.deceased_id} - {self.date_of_death}>'
