from extensions import db 
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from departments.models.user import User
class SOAPNote(db.Model):
    __tablename__ = 'soap_notes'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Text, nullable=False)  # Link to a patient model if available
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # SOAPNote Fields
    situation = db.Column(db.Text, nullable=False)
    hpi = db.Column(db.Text, nullable=False)
    aggravating_factors = db.Column(db.Text)
    alleviating_factors = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    medication_history = db.Column(db.Text)
    assessment = db.Column(db.Text, nullable=False)
    recommendation = db.Column(db.Text, nullable=True)
    additional_notes = db.Column(db.Text)
    symptoms = db.Column(db.Text, nullable=True)
    ai_notes= db.Column(db.Text)
    ai_analysis = db.Column(db.Text)  # New column

    # File Upload Path (if any)
    file_path = db.Column(db.String(255))  # Store the path to the uploaded file

    def __repr__(self):
        return f"<SOAPNote {self.id} - Patient {self.patient_id}>"
class Medicine(db.Model):
    __tablename__ = 'medicines'
    id = db.Column(db.Integer, primary_key=True)
    generic_name = db.Column(db.String(100), nullable=False)
    brand_name = db.Column(db.String(100), nullable=False)
    dosage = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<Medicine {self.generic_name} ({self.brand_name})>"


class PrescribedMedicine(db.Model):
    __tablename__ = 'prescribed_medicines'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Text, db.ForeignKey('patients.patient_id'), nullable=False)
    medicine_id = db.Column(db.Integer, db.ForeignKey('medicines.id'), nullable=False)
    dosage = db.Column(db.String(255), nullable=False)
    strength = db.Column(db.String(255), nullable=False)
    frequency = db.Column(db.String(255), nullable=False)
    prescription_id = db.Column(db.String(36), nullable=False)  # UUID as string
    num_days = db.Column(db.Integer, nullable=False)
    status = db.Column(db.Integer, default=0, nullable=False)

    # Define relationship with Medicine
    medicine = relationship("Medicine", backref="prescribed_medicines")

    def __init__(self, patient_id, medicine_id, dosage, strength, frequency, prescription_id, num_days):
        self.patient_id = patient_id
        self.medicine_id = medicine_id
        self.dosage = dosage
        self.strength = strength
        self.frequency = frequency
        self.prescription_id = prescription_id
        self.num_days = num_days

    def __repr__(self):
        return f'<PrescribedMedicine {self.medicine_id} for Patient {self.patient_id}>'


class LabTest(db.Model):
    __tablename__ = 'labtests'

    id = db.Column(db.Integer, primary_key=True)
    test_name = db.Column(db.String(100), nullable=False)
    cost = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)

    def __repr__(self):
        return f"<LabTest {self.test_name}>"




class RequestedLab(db.Model):
    __tablename__ = 'requested_labs'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.patient_id'), nullable=False)
    lab_test_id = db.Column(db.Integer, db.ForeignKey('labtests.id'), nullable=False)
    date_requested = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.Integer, default=0)
    receipt_number = db.Column(db.String(255), nullable=True)  # Billing integration
    description = db.Column(db.Text, nullable=True)            # New column
    result_id = db.Column(db.String(255), nullable=True)       # New column

    patient = db.relationship('Patient', backref=db.backref('requested_labs', lazy=True))
    lab_test = db.relationship('LabTest', backref=db.backref('requested_labs', lazy=True))

    def __repr__(self):
        return f'<RequestedLab {self.lab_test_id} for Patient {self.patient_id}>'


# departments/models/imaging.py
class ImagingResult(db.Model):
    __tablename__ = 'imaging_results'
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.String, unique=True, nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    imaging_id = db.Column(db.Integer, db.ForeignKey('imaging.id'), nullable=False)
    test_date = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    result_notes = db.Column(db.Text)
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    dicom_file_path = db.Column(db.String, nullable=True)
    ai_findings = db.Column(db.Text, nullable=True)
    ai_generated = db.Column(db.Boolean, default=False, nullable=False)
    files_processed = db.Column(db.Integer, default=0, nullable=False)
    files_failed = db.Column(db.Integer, default=0, nullable=False)
    processing_metadata = db.Column(db.JSON, nullable=True)

    # Relationships
    patient = db.relationship('Patient', backref='imaging_results')
    imaging = db.relationship('Imaging', backref='imaging_results')
    updated_by_user = db.relationship('User', backref='imaging_results_updated')
    requested_image = db.relationship('RequestedImage', backref='imaging_result', uselist=False)

    def get_file_count(self):
        """Return the total number of files processed and failed."""
        return self.files_processed + self.files_failed

    def __repr__(self):
        return f"<ImagingResult {self.result_id} - Patient {self.patient_id}, Imaging {self.imaging_id}>"


# Imaging Tests Table
class Imaging(db.Model):
    __tablename__ = 'imaging'
    id = db.Column(db.Integer, primary_key=True)
    imaging_type = db.Column(db.String(100), nullable=False)
    cost = db.Column(db.Float, nullable=False)

    # Relationship to RequestedImage
    requested_images = db.relationship('RequestedImage', backref='imaging', lazy=True)

    def __repr__(self):
        return f"<Imaging {self.imaging_type}>"    


# Imaging Requests Table
class RequestedImage(db.Model):
    __tablename__ = 'requested_images'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Text, db.ForeignKey('patients.patient_id'), nullable=False)
    imaging_id = db.Column(db.Integer, db.ForeignKey('imaging.id'), nullable=False)
    date_requested = db.Column(db.DateTime, default=db.func.current_timestamp(), nullable=False)
    status = db.Column(db.Integer, default=0)  # 0: Pending, 1: Processed, 2: Cancelled
    result_id = db.Column(db.String, db.ForeignKey('imaging_results.result_id'), unique=True, nullable=True)
    description = db.Column(db.Text, nullable=True)
    receipt_number = db.Column(db.String(255), nullable=True)  # Added for billing integration

    # Relationships
    patient = db.relationship('Patient', backref='requested_images')

    def __repr__(self):
        return f"<RequestedImage {self.imaging_id} for Patient {self.patient_id}>"

#unmatched requests
class UnmatchedImagingRequest(db.Model):
    __tablename__ = 'unmatched_imaging_requests'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.patient_id'), nullable=False)
    description = db.Column(db.Text, nullable=False)
    date_requested = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship('Patient', backref=db.backref('unmatched_imaging_requests', lazy=True))

    def __repr__(self):
        return f"<UnmatchedImagingRequest for Patient {self.patient_id}: {self.description}>"
    
#table theater procedures
class TheatreProcedure(db.Model):
    __tablename__ = 'theatre_procedures'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(100))
    cost = db.Column(db.Numeric(10, 2), nullable=False)
    created_by = db.Column(db.Integer)
    updated_by = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<TheatreProcedure(id={self.id}, name={self.name}, type={self.type}, cost={self.cost})>"   
class TheatreList(db.Model):
    __tablename__ = 'theatre_list'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey('patients.patient_id'), nullable=False)  # Changed to String
    procedure_id = db.Column(db.Integer, db.ForeignKey('theatre_procedures.id'), nullable=False)
    status = db.Column(db.Integer, default=0)
    created_by = db.Column(db.Integer)
    done_by = db.Column(db.Integer)
    notes_on_book = db.Column(db.Text)
    notes_on_post_op = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    receipt_number = db.Column(db.String(255), nullable=True)  # Added for billing integration


    # Relationships
    patient = db.relationship('Patient', backref='theatre_entries')
    procedure = db.relationship('TheatreProcedure', backref='theatre_procedures')

    def __repr__(self):
        return f"<TheatreList(id={self.id}, patient_id={self.patient_id}, procedure={self.procedure_id}, status={self.status})>"  

class Ward(db.Model):
    """Model for hospital wards."""
    __tablename__ = "wards"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    sex = db.Column(db.String(10), nullable=False)  # 'Male', 'Female', 'Mixed'
    number_of_beds = db.Column(db.Integer, nullable=False, default=0)
    occupied_beds = db.Column(db.Integer, nullable=False, default=0)  # New field
    daily_charge = db.Column(db.Numeric(10, 2), nullable=False)

    def available_beds(self):
        """Returns the number of available beds in the ward."""
        return self.number_of_beds - self.occupied_beds

    def __repr__(self):
        return f"<Ward {self.name} - {self.sex} - Beds: {self.number_of_beds} - Occupied: {self.occupied_beds}>"


class AdmittedPatient(db.Model):
    """Model for tracking admitted patients."""
    __tablename__ = "admitted_patients"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(20), db.ForeignKey("patients.patient_id"), nullable=False)
    ward_id = db.Column(db.Integer, db.ForeignKey("wards.id"), nullable=False)
    admitted_on = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    admission_criteria = db.Column(db.Text, nullable=False)
    admitted_by = db.Column(db.Integer, nullable=False)  # User ID of admitting doctor/staff
    discharged_on = db.Column(db.DateTime, nullable=True)  # NULL if still admitted
    discharge_summary = db.Column(db.Text, nullable=True)
    prescribed_drugs = db.Column(db.Text, nullable=True)
    follow_up_clinic = db.Column(db.String(255), nullable=True)
    receipt_number = db.Column(db.String(255), nullable=True)  # Added for billing integration

    # Relationships
    ward = db.relationship("Ward", backref="admitted_patients")
    
    def __repr__(self):
        return f"<AdmittedPatient {self.patient_id} - Ward: {self.ward_id} - Admitted On: {self.admitted_on}>"  

class WardBedHistory(db.Model):
    """Tracks changes in bed usage for wards."""
    __tablename__ = "ward_bed_history"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ward_id = db.Column(db.Integer, db.ForeignKey("wards.id"), nullable=False)
    patient_id = db.Column(db.String(20), db.ForeignKey("patients.patient_id"), nullable=False)
    action = db.Column(db.String(10), nullable=False)  # 'Admit' or 'Discharge'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    ward = db.relationship("Ward", backref="bed_history")
    patient = db.relationship("Patient", backref="bed_history")

    def __repr__(self):
        return f"<BedHistory {self.action} - Ward: {self.ward_id} - Patient: {self.patient_id}>"
    
class WardRoom(db.Model):
    """Model to track individual rooms in a ward."""
    __tablename__ = "ward_rooms"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ward_id = db.Column(db.Integer, db.ForeignKey("wards.id"), nullable=False)
    room_number = db.Column(db.String(20), nullable=False)
    occupied = db.Column(db.Boolean, default=False, nullable=False)  # Tracks availability

    # Relationships
    ward = db.relationship("Ward", backref="rooms")

    def __repr__(self):
        return f"<Room {self.room_number} - Ward {self.ward_id} - {'Occupied' if self.occupied else 'Available'}>"
class Bed(db.Model):
    """Model to track individual beds within a room."""
    __tablename__ = "beds"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    room_id = db.Column(db.Integer, db.ForeignKey("ward_rooms.id"), nullable=False)
    bed_number = db.Column(db.String(20), nullable=False)
    occupied = db.Column(db.Boolean, default=False, nullable=False)  # Track availability

    # Relationships
    room = db.relationship("WardRoom", backref="beds")

    def __repr__(self):
        return f"<Bed {self.bed_number} - Room {self.room_id} - {'Occupied' if self.occupied else 'Available'}>"
class WardRound(db.Model):
    __tablename__ = 'ward_rounds'

    id = db.Column(db.Integer, primary_key=True)
    admission_id = db.Column(db.Integer, db.ForeignKey('admitted_patients.id'), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    notes = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default="Under Treatment")  # Example: "Ready for Discharge", "Critical", etc.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    doctor = db.relationship("User", backref="ward_rounds")    


class Disease(db.Model):
    __tablename__ = 'diseases'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    cui = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)

    keywords = db.relationship('DiseaseKeyword', backref='disease', lazy=True)
    symptoms = db.relationship('Symptom', secondary='disease_symptoms', back_populates='diseases')
    management_plan = db.relationship('DiseaseManagementPlan', backref='disease', uselist=False, lazy=True)
    
    # New relationship for lab tests
    lab_tests = db.relationship('DiseaseLab', backref='disease', lazy=True)


class Symptom(db.Model):
    __tablename__ = 'symptoms'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    cui = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text)

    diseases = db.relationship('Disease', secondary='disease_symptoms', back_populates='symptoms')


class DiseaseKeyword(db.Model):
    __tablename__ = 'disease_keywords'
    id = db.Column(db.Integer, primary_key=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)
    keyword = db.Column(db.String(255), nullable=False)
    cui = db.Column(db.String(50))


class DiseaseManagementPlan(db.Model):
    __tablename__ = 'disease_management_plans'
    id = db.Column(db.Integer, primary_key=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)
    plan = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class DiseaseSymptom(db.Model):
    __tablename__ = 'disease_symptoms'
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), primary_key=True)
    symptom_id = db.Column(db.Integer, db.ForeignKey('symptoms.id'), primary_key=True)


# New Model: DiseaseLab
class DiseaseLab(db.Model):
    __tablename__ = 'disease_labs'
    id = db.Column(db.Integer, primary_key=True)
    disease_id = db.Column(db.Integer, db.ForeignKey('diseases.id'), nullable=False)
    lab_test = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)

    # Relationship back to Disease handled via backref

  
