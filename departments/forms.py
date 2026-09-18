from datetime import date

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    DateField,
    FloatField,
    HiddenField,
    IntegerField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import DataRequired, Email, Length, Optional, ValidationError

from departments.models.medicine import CancerStage, CancerType
from departments.models.records import Patient


class AdmitPatientForm(FlaskForm):
    patient_id = SelectField("Patient", choices=[], validators=[DataRequired()])
    ward_id = SelectField("Ward", choices=[], validators=[DataRequired()])
    room_id = SelectField("Room", choices=[], validators=[DataRequired()])
    bed_id = SelectField("Bed", choices=[], validators=[DataRequired()])
    admission_criteria = TextAreaField(
        "Admission Criteria", validators=[DataRequired()]
    )
    admitted_by = HiddenField()


class AddDeductionForm(FlaskForm):
    name = StringField("Deduction Name", validators=[DataRequired()])
    value = FloatField("Value", validators=[DataRequired()])
    is_percentage = BooleanField("Is Percentage?")
    submit = SubmitField("Add Deduction")


class AddAllowanceForm(FlaskForm):
    job_group = StringField("Job Group", validators=[DataRequired()])
    name = StringField("Allowance Name", validators=[DataRequired()])
    value = FloatField("Value", validators=[DataRequired()])
    submit = SubmitField("Add Allowance")


class LeaveRequestForm(FlaskForm):
    start_date = DateField("Start Date", validators=[DataRequired()])
    end_date = DateField("End Date", validators=[DataRequired()])
    type = SelectField(
        "Leave Type",
        choices=[
            ("annual", "Annual Leave"),
            ("vacation", "Vacation Leave"),
            ("sick", "Sick Leave"),
            ("maternity", "Maternity Leave"),
            ("paternity", "Paternity Leave"),
            ("compassionate", "Compassionate Leave"),
            ("study", "Study / Exam Leave"),
            ("unpaid", "Unpaid Leave"),
        ],
        validators=[DataRequired()],
    )
    submit = SubmitField("Submit Leave Request")

    def validate_end_date(self, field):
        """
        Previously nothing checked date ordering at all, so a request with
        end_date before start_date was accepted as-is: Leave.days (and any
        balance math built on it) would silently go negative instead of
        the request being rejected up front.
        """
        if self.start_date.data and field.data and field.data < self.start_date.data:
            raise ValidationError("End date cannot be before the start date.")


class UpdateProfileForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Email()])
    phone = StringField("Phone", validators=[DataRequired()])
    bank_name = StringField("Bank Name", validators=[DataRequired()])
    bank_account = StringField("Bank Account", validators=[DataRequired()])
    submit = SubmitField("Update Profile")


class PatientSearchForm(FlaskForm):
    patient_id = SelectField(
        "Patient", validators=[DataRequired(message="Please select a patient.")]
    )
    submit_search = SubmitField("Search")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patient_id.choices = [
            (p.patient_id, f"{p.name} ({p.patient_id})")
            for p in Patient.query.order_by(Patient.name).all()
        ]


class OncoPatientForm(FlaskForm):
    diagnosis = StringField("Diagnosis", validators=[DataRequired(), Length(max=200)])
    diagnosis_date = DateField(
        "Diagnosis Date", validators=[DataRequired()], format="%Y-%m-%d"
    )

    # Will be populated in __init__ from database
    cancer_type = SelectField(
        "Cancer Type", choices=[], coerce=int, validators=[DataRequired()]
    )
    stage = SelectField("Stage", choices=[], coerce=int, validators=[DataRequired()])

    status = SelectField(
        "Status",
        choices=[
            ("Active", "Active"),
            ("Completed", "Completed"),
            ("Discontinued", "Discontinued"),
        ],
        validators=[DataRequired()],
    )
    submit_update = SubmitField("Update Oncology Details")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load cancer types from DB: [(id, name), ...]
        self.cancer_type.choices = [
            (ct.id, ct.name) for ct in CancerType.query.order_by(CancerType.name).all()
        ]

        # Load all stages initially (can be filtered later by JS)
        self.stage.choices = [
            (st.id, st.label) for st in CancerStage.query.order_by(CancerStage.id).all()
        ]


class OncologyNoteForm(FlaskForm):
    note_date = DateField(
        "Note Date", validators=[DataRequired()], format="%Y-%m-%d", default=date.today
    )
    note_content = TextAreaField(
        "Note Content", validators=[DataRequired(), Length(min=1, max=1000)]
    )
    submit_note = SubmitField("Add Note")

class PerformanceReviewForm(FlaskForm):
    review_period = StringField("Review Period (e.g. 2026-Annual)", validators=[DataRequired()])
    review_type = SelectField(
        "Review Type",
        choices=[("annual", "Annual"), ("mid-year", "Mid-Year"), ("probation", "Probation")],
    )
    score = SelectField(
        "Overall Score",
        choices=[
            ("1", "1 – Unsatisfactory"),
            ("2", "2 – Below Expectations"),
            ("3", "3 – Meets Expectations"),
            ("4", "4 – Exceeds Expectations"),
            ("5", "5 – Outstanding"),
        ],
        coerce=int,
    )
    strengths = TextAreaField("Strengths")
    areas_for_improvement = TextAreaField("Areas for Improvement")
    goals_next_period = TextAreaField("Goals for Next Period")
    comments = TextAreaField("Additional Comments")
    submit = SubmitField("Save Review")


class TrainingRecordForm(FlaskForm):
    title = StringField("Training Title", validators=[DataRequired()])
    provider = StringField("Provider / Institution")
    training_type = SelectField(
        "Training Type",
        choices=[
            ("cpd", "CPD / Continuous Professional Development"),
            ("mandatory", "Mandatory / Statutory"),
            ("skills", "Clinical / Technical Skills"),
            ("leadership", "Leadership & Management"),
            ("induction", "Induction"),
            ("conference", "Conference / Seminar"),
        ],
    )
    date_completed = DateField("Date Completed", validators=[DataRequired()])
    expiry_date = DateField("Expiry Date (if applicable)", validators=[Optional()])
    cpd_points = IntegerField("CPD Points Earned", validators=[Optional()])
    certificate_number = StringField("Certificate / Reference Number")
    notes = TextAreaField("Notes")
    submit = SubmitField("Save Record")


class DisciplinaryRecordForm(FlaskForm):
    incident_date = DateField("Incident Date", validators=[DataRequired()])
    incident_type = SelectField(
        "Incident / Action Type",
        choices=[
            ("verbal_warning", "Verbal Warning"),
            ("written_warning", "Written Warning"),
            ("final_warning", "Final Written Warning"),
            ("suspension", "Suspension"),
            ("dismissal", "Dismissal"),
            ("other", "Other"),
        ],
    )
    description = TextAreaField("Incident Description", validators=[DataRequired()])
    action_taken = TextAreaField("Action Taken", validators=[DataRequired()])
    outcome = SelectField(
        "Outcome",
        choices=[
            ("", "— select —"),
            ("resolved", "Resolved"),
            ("appeal_pending", "Appeal Pending"),
            ("dismissed", "Dismissed"),
        ],
        validators=[Optional()],
    )
    submit = SubmitField("Save Record")
