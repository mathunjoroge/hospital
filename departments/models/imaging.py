from extensions import db


# Imaging Results Table
class ImagingResult(db.Model):
    __tablename__ = "imaging_results"
    id = db.Column(db.Integer, primary_key=True)
    result_id = db.Column(db.String, unique=True, nullable=False)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    imaging_id = db.Column(db.Integer, db.ForeignKey("imaging.id"), nullable=False)
    test_date = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )
    result_notes = db.Column(db.Text)
    updated_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    dicom_file_path = db.Column(db.String, nullable=True)
    ai_findings = db.Column(db.Text, nullable=True)
    ai_generated = db.Column(db.Boolean, default=False, nullable=False)
    files_processed = db.Column(db.Integer, default=0, nullable=False)
    files_failed = db.Column(db.Integer, default=0, nullable=False)
    processing_metadata = db.Column(db.JSON, nullable=True)

    # Relationships
    patient = db.relationship("Patient", backref="imaging_results")
    imaging = db.relationship("Imaging", backref="imaging_results")
    updated_by_user = db.relationship("User", backref="imaging_results_updated")
    requested_image = db.relationship(
        "RequestedImage", backref="imaging_result", uselist=False
    )

    def get_file_count(self):
        """Return the total number of files processed and failed."""
        return self.files_processed + self.files_failed

    def __repr__(self):
        return f"<ImagingResult {self.result_id} - Patient {self.patient_id}, Imaging {self.imaging_id}>"
