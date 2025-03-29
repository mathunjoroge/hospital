from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, BooleanField, SubmitField, DateField, SelectField
from wtforms.validators import DataRequired, Email
from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, TextAreaField, HiddenField
from wtforms.validators import DataRequired

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
class AdmitPatientForm(FlaskForm):
    patient_id = SelectField('Patient', choices=[], validators=[DataRequired()])
    ward_id = SelectField('Ward', choices=[], validators=[DataRequired()])
    room_id = SelectField('Room', choices=[], validators=[DataRequired()])
    bed_id = SelectField('Bed', choices=[], validators=[DataRequired()])
    admission_criteria = TextAreaField('Admission Criteria', validators=[DataRequired()])
    admitted_by = HiddenField() 
