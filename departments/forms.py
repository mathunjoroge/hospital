from flask_wtf import FlaskForm
from wtforms import (
    StringField,
    FloatField,
    BooleanField,
    SubmitField,
    DateField,
    SelectField,
    TextAreaField,
    HiddenField
)
from wtforms.validators import DataRequired, Email, Length
from departments.models.records import Patient
from departments.models.medicine import CancerType, CancerStage, CancerTypeStage
from datetime import datetime

class AdmitPatientForm(FlaskForm):
    patient_id = SelectField('Patient', choices=[], validators=[DataRequired()])
    ward_id = SelectField('Ward', choices=[], validators=[DataRequired()])
    room_id = SelectField('Room', choices=[], validators=[DataRequired()])
    bed_id = SelectField('Bed', choices=[], validators=[DataRequired()])
    admission_criteria = TextAreaField('Admission Criteria', validators=[DataRequired()])
    admitted_by = HiddenField() 
class AddDeductionForm(FlaskForm):
    name = StringField('Deduction Name', validators=[DataRequired()])
    value = FloatField('Value', validators=[DataRequired()])
    is_percentage = BooleanField('Is Percentage?')
    submit = SubmitField('Add Deduction')

class AddAllowanceForm(FlaskForm):
    job_group = StringField('Job Group', validators=[DataRequired()])
    name = StringField('Allowance Name', validators=[DataRequired()])
    value = FloatField('Value', validators=[DataRequired()])
    submit = SubmitField('Add Allowance')

class LeaveRequestForm(FlaskForm):
    start_date = DateField('Start Date', validators=[DataRequired()])
    end_date = DateField('End Date', validators=[DataRequired()])
    type = SelectField('Leave Type', choices=[('sick', 'Sick Leave'), ('vacation', 'Vacation')], validators=[DataRequired()])
    submit = SubmitField('Submit Leave Request')

class UpdateProfileForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone', validators=[DataRequired()])
    bank_name = StringField('Bank Name', validators=[DataRequired()])
    bank_account = StringField('Bank Account', validators=[DataRequired()])
    submit = SubmitField('Update Profile')    
class PatientSearchForm(FlaskForm):
    patient_id = SelectField('Patient', validators=[DataRequired(message="Please select a patient.")])
    submit_search = SubmitField('Search')

    def __init__(self, *args, **kwargs):
        super(PatientSearchForm, self).__init__(*args, **kwargs)
        self.patient_id.choices = [(p.patient_id, f"{p.name} ({p.patient_id})") for p in Patient.query.order_by(Patient.name).all()]


class OncoPatientForm(FlaskForm):
    diagnosis = StringField('Diagnosis', validators=[DataRequired(), Length(max=200)])
    diagnosis_date = DateField('Diagnosis Date', validators=[DataRequired()], format='%Y-%m-%d')

    # Will be populated in __init__ from database
    cancer_type = SelectField('Cancer Type', choices=[], coerce=int, validators=[DataRequired()])
    stage = SelectField('Stage', choices=[], coerce=int, validators=[DataRequired()])

    status = SelectField(
        'Status',
        choices=[
            ('Active', 'Active'),
            ('Completed', 'Completed'),
            ('Discontinued', 'Discontinued')
        ],
        validators=[DataRequired()]
    )
    submit_update = SubmitField('Update Oncology Details')

    def __init__(self, *args, **kwargs):
        super(OncoPatientForm, self).__init__(*args, **kwargs)

        # Load cancer types from DB: [(id, name), ...]
        self.cancer_type.choices = [
            (ct.id, ct.name) for ct in CancerType.query.order_by(CancerType.name).all()
        ]

        # Load all stages initially (can be filtered later by JS)
        self.stage.choices = [
            (st.id, st.label) for st in CancerStage.query.order_by(CancerStage.id).all()
        ]
class OncologyNoteForm(FlaskForm):
    note_date = DateField('Note Date', validators=[DataRequired()], format='%Y-%m-%d', default=datetime.utcnow().date)
    note_content = TextAreaField('Note Content', validators=[DataRequired(), Length(min=1, max=1000)])
    submit_note = SubmitField('Add Note')
  
