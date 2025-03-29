from extensions import db
from datetime import datetime

class NursingNote(db.Model):
    __tablename__ = 'nursing_notes'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False)
    nurse_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    note = db.Column(db.Text, nullable=False)  # General observations
    allergies = db.Column(db.String(255), nullable=True)  # e.g., "Penicillin, Nuts"
    code_status = db.Column(db.String(20), nullable=True)  # e.g., "Full Code"
    medications = db.Column(db.Text, nullable=True)  # JSON or delimited string: "Aspirin|81mg|PO|08:00"
    shift_update = db.Column(db.Text, nullable=True)  # e.g., "Patient stable, BP checked"
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # Add relationship
    nurse = db.relationship('User', backref='nursing_notes')

class NursingCareTask(db.Model):
    __tablename__ = 'nursing_care_tasks'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False)
    nurse_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    task_description = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='Pending')
    priority = db.Column(db.String(20), nullable=False, default='Medium')  # e.g., Low, Medium, High
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)
    nurse = db.relationship('User', backref='care_tasks')

class Vitals(db.Model):
    __tablename__ = 'vitals'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), nullable=False)  # Adjusted to match NursingNote
    nurse_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    temperature = db.Column(db.Float, nullable=True)  # Celsius
    pulse = db.Column(db.Integer, nullable=True)  # Beats per minute
    blood_pressure_systolic = db.Column(db.Integer, nullable=True)  # mmHg
    blood_pressure_diastolic = db.Column(db.Integer, nullable=True)  # mmHg
    respiratory_rate = db.Column(db.Integer, nullable=True)  # Breaths per minute
    oxygen_saturation = db.Column(db.Integer, nullable=True)  # Percentage (SpO₂)
    blood_glucose = db.Column(db.Float, nullable=True)  # mg/dL
    weight = db.Column(db.Float, nullable=True)  # kg
    height = db.Column(db.Float, nullable=True)  # cm
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    # Add relationship
    nurse = db.relationship('User', backref='vitals')

class Partogram(db.Model):
    __tablename__ = 'partogram'

    # Primary key
    id = db.Column(db.Integer, primary_key=True)

    # Patient and metadata
    patient_id = db.Column(db.String(50), nullable=False, index=True)  # To group entries for the same patient
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())  # When the entry was recorded
    recorded_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # ID of the user who recorded it
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Add relationship to User
    recorded_by_user = db.relationship('User', backref='partograms')

    # Fetal Conditions (Top Section of Partograph)
    fetal_heart_rate = db.Column(db.Integer, nullable=False)  # In beats per minute (bpm)
    amniotic_fluid = db.Column(db.String(20), nullable=False)  # e.g., 'clear', 'meconium', 'blood', 'absent'
    moulding = db.Column(db.String(10), nullable=False)  # e.g., 'none', '1+', '2+', '3+'

    # Progress of Labour (Middle Section of Partograph)
    cervical_dilation = db.Column(db.Float, nullable=False)  # In cm (0-10)
    time_hours = db.Column(db.Float, nullable=False)  # Hours since active labor start
    labour_status = db.Column(db.String(50), nullable=False)  # e.g., 'Normal', 'Crossed Alert Line', 'Crossed Action Line'

    # Maternal Conditions (Bottom Section of Partograph)
    contractions = db.Column(db.Integer, nullable=False)  # Number of contractions per 10 minutes
    oxytocin = db.Column(db.Float, nullable=True)  # Units of oxytocin administered
    drugs = db.Column(db.String(100), nullable=True)  # Any drugs given (e.g., 'Paracetamol')
    pulse = db.Column(db.Integer, nullable=False)  # Maternal pulse in bpm
    bp_systolic = db.Column(db.Integer, nullable=False)  # Systolic blood pressure in mmHg
    bp_diastolic = db.Column(db.Integer, nullable=False)  # Diastolic blood pressure in mmHg
    temperature = db.Column(db.Float, nullable=False)  # Maternal temperature in °C
    urine_protein = db.Column(db.String(10), nullable=False)  # e.g., 'negative', '1+', '2+', '3+'
    urine_volume = db.Column(db.Integer, nullable=False)  # In mL
    urine_acetone = db.Column(db.String(10), nullable=False)  # e.g., 'negative', '1+', '2+', '3+'
    # Patient and metadata
   
    timestamp = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp(), index=True)

    def __repr__(self):
        return f"<Partogram(id={self.id}, patient_id={self.patient_id}, timestamp={self.timestamp})>"

# New models for medication_admin, messages, and notifications
class MedicationAdmin(db.Model):
    __tablename__ = 'medication_admin'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    medication = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(50), nullable=False)
    time_administered = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    user = db.relationship('User', backref='medication_admin')

class Messages(db.Model):
    __tablename__ = 'messages'
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    patient_id = db.Column(db.String(50), nullable=True)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    receiver = db.relationship('User', foreign_keys=[receiver_id], backref='received_messages')

class Notifications(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    receiver = db.relationship('User', backref='notifications')