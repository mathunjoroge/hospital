from extensions import db


class LabResultTemplate(db.Model):
    __tablename__ = "labresults_templates"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("labtests.id"), nullable=False)
    parameter_name = db.Column(db.String(255), nullable=False)
    normal_range_low = db.Column(db.Float, nullable=False)
    normal_range_high = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(50), nullable=False)
    # Relationship to LabTest
    lab_test = db.relationship("LabTest", backref="result_templates")

    def __repr__(self):
        return f"<LabResultTemplate {self.parameter_name} for Test {self.test_id}>"


class LabResult(db.Model):
    __tablename__ = "lab_results"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    lab_test_id = db.Column(db.Integer, db.ForeignKey("labtests.id"), nullable=False)
    test_date = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )
    result_notes = db.Column(db.Text)  # Optional notes

    # Newly added columns
    result_id = db.Column(
        db.String, unique=True, nullable=False
    )  # Unique identifier for result
    result = db.Column(db.Text, nullable=True)  # Stores the test result
    updated_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )  # Who updated the result
    status = db.Column(db.String(30), default="PENDING_VERIFICATION")
    panic_status = db.Column(db.String(30), default="NORMAL")
    panic_message = db.Column(db.Text, nullable=True)
    verified_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    verified_at = db.Column(db.DateTime, nullable=True)

    # Phase 4 — HL7v2 MLLP Interface (additive, no existing columns renamed/dropped)
    source_system = db.Column(
        db.String(80), nullable=True, default=None
    )  # e.g. "HL7_LIS", "FHIR_SOURCE", "JSON_LIS"
    raw_hl7 = db.Column(
        db.Text, nullable=True, default=None
    )  # Raw HL7v2 ER7 text for audit trail

    # Relationships (if needed)
    patient = db.relationship("Patient", backref=db.backref("lab_results", lazy=True))
    lab_test = db.relationship("LabTest", backref=db.backref("lab_results", lazy=True))
    updated_by_user = db.relationship(
        "User",
        foreign_keys=[updated_by],
        backref=db.backref("updated_results", lazy=True),
    )
    verifier = db.relationship(
        "User",
        foreign_keys=[verified_by],
        backref=db.backref("verified_results", lazy=True),
    )

    def __repr__(self):
        return f"<LabResult {self.id} - Patient {self.patient_id}, Test {self.lab_test_id}>"


class Specimen(db.Model):
    __tablename__ = "specimens"

    id = db.Column(db.Integer, primary_key=True)
    barcode = db.Column(db.String(50), unique=True, index=True, nullable=False)
    requested_lab_id = db.Column(
        db.Integer, db.ForeignKey("requested_labs.id"), nullable=True
    )
    patient_id = db.Column(
        db.String(20), db.ForeignKey("patients.patient_id"), nullable=False
    )
    specimen_type = db.Column(db.String(50), nullable=False, default="WHOLE_BLOOD")
    container_type = db.Column(db.String(50), nullable=False, default="EDTA_PURPLE")
    status = db.Column(db.String(30), nullable=False, default="ORDERED")
    collected_by_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )
    collected_at = db.Column(db.DateTime, nullable=True)
    received_at = db.Column(db.DateTime, nullable=True)
    rejection_reason = db.Column(db.String(100), nullable=True)
    chain_of_custody = db.Column(db.Text, nullable=True)  # JSON string
    created_at = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )

    patient = db.relationship("Patient", backref=db.backref("specimens", lazy=True))
    requested_lab = db.relationship(
        "RequestedLab", backref=db.backref("specimens", lazy=True)
    )
    collector = db.relationship("User", foreign_keys=[collected_by_id])

    def __repr__(self):
        return f"<Specimen {self.barcode} - Status {self.status}>"


class LabQCSample(db.Model):
    __tablename__ = "lab_qc_samples"

    id = db.Column(db.Integer, primary_key=True)
    control_name = db.Column(db.String(100), nullable=False)
    lot_number = db.Column(db.String(50), nullable=False)
    analyzer_name = db.Column(db.String(100), nullable=False)
    parameter_name = db.Column(db.String(100), nullable=False)
    target_mean = db.Column(db.Float, nullable=False)
    target_sd = db.Column(db.Float, nullable=False)
    expiration_date = db.Column(db.Date, nullable=True)
    created_at = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )

    def __repr__(self):
        return f"<LabQCSample {self.control_name} - {self.parameter_name}>"


class LabQCResult(db.Model):
    __tablename__ = "lab_qc_results"

    id = db.Column(db.Integer, primary_key=True)
    qc_sample_id = db.Column(
        db.Integer, db.ForeignKey("lab_qc_samples.id"), nullable=False
    )
    run_timestamp = db.Column(
        db.DateTime, default=db.func.current_timestamp(), nullable=False
    )
    measured_value = db.Column(db.Float, nullable=False)
    z_score = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False, default="PASS")
    violated_rules = db.Column(db.Text, nullable=True)  # JSON string
    operator_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)

    qc_sample = db.relationship(
        "LabQCSample",
        backref=db.backref("qc_results", lazy=True, cascade="all, delete-orphan"),
    )
    operator = db.relationship("User", foreign_keys=[operator_id])

    def __repr__(self):
        return f"<LabQCResult ID={self.id} Z={self.z_score:.2f} Status={self.status}>"

