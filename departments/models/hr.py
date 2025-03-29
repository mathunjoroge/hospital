from extensions import db
from sqlalchemy.orm import relationship
from datetime import datetime

# Rota Table
class Rota(db.Model):
    """Represents an employee rota."""
    __tablename__ = 'rotas'

    id = db.Column(db.Integer, primary_key=True)
    week_range = db.Column(db.String(20), nullable=False)  # Week range (e.g., "01/03/2023 - 07/03/2023")
    shift_8_5 = db.Column(db.Text, nullable=True)  # Morning shift (comma-separated names)
    shift_5_8 = db.Column(db.String(100), nullable=True)  # Evening shift
    shift_8_8 = db.Column(db.String(100), nullable=True)  # Night shift

    def __repr__(self):
        return f"<Rota {self.week_range}>"

# Employee Table
class Employee(db.Model):
    """Represents an employee in the HR department."""
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(20), unique=True, nullable=False)  # Unique employee ID (e.g., E0001)
    name = db.Column(db.String(100), nullable=False)                    # Employee's full name
    role = db.Column(db.String(50), nullable=False)                      # Role/privileges (e.g., records, nursing)
    department = db.Column(db.String(50), nullable=False)                # Department assigned to
    job_group = db.Column(db.String(50), nullable=False)                 # Job group (e.g., "Group A", "Group B")
    date_hired = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # Hire date
    updated_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Updated by user
    is_active = db.Column(db.Boolean, default=True, nullable=False)      # Employment status
    email = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    bank_name = db.Column(db.String(100), nullable=True)
    bank_account = db.Column(db.String(50), nullable=True)

    def __repr__(self):
        return f"<Employee {self.name} - ID: {self.employee_id}>"
    
class Allowance(db.Model):
    """Represents an allowance for a specific job group."""
    __tablename__ = 'allowances'

    id = db.Column(db.Integer, primary_key=True)
    job_group = db.Column(db.String(50), nullable=False)  # Job group (e.g., "Group A", "Group B")
    name = db.Column(db.String(100), nullable=False)      # Allowance name (e.g., "Housing Allowance")
    value = db.Column(db.Float, nullable=False)           # Absolute allowance value

    def __repr__(self):
        return f"<Allowance {self.name} - Job Group: {self.job_group}, Value: {self.value}>"

class Payroll(db.Model):
    """Represents a payroll record for an employee."""
    __tablename__ = 'payrolls'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
    month = db.Column(db.String(20), nullable=False)      # Payroll month (e.g., "January 2023")
    gross_pay = db.Column(db.Float, nullable=False)       # Total earnings (basic salary + allowances)
    total_deductions = db.Column(db.Float, nullable=False)  # Total deductions
    net_pay = db.Column(db.Float, nullable=False)        # Net pay (gross_pay - total_deductions)

    employee = db.relationship('Employee', backref='payrolls')

    def __repr__(self):
        return f"<Payroll {self.month} - Employee: {self.employee.name}, Net Pay: {self.net_pay}>"

class Deduction(db.Model):
    """Represents a deduction (e.g., PAYE, NHIF, NSSF)."""
    __tablename__ = 'deductions'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)      # Deduction name (e.g., "PAYE")
    value = db.Column(db.Float, nullable=False)           # Deduction value (fixed or percentage)
    is_percentage = db.Column(db.Boolean, default=False)  # Whether the value is a percentage

    def __repr__(self):
        return f"<Deduction {self.name} - Value: {self.value}>"    
class Leave(db.Model):
    __tablename__ = 'leaves'
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    type = db.Column(db.String(50), nullable=False)  # e.g., sick leave, vacation
    status = db.Column(db.String(20), default='Pending')  # e.g., Pending, Approved, Rejected  
class CustomRule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=True)
    job_group = db.Column(db.String(50), nullable=True)
    type = db.Column(db.String(50), nullable=False)  # e.g., deduction, allowance
    name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    is_percentage = db.Column(db.Boolean, default=False)   
class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)  # e.g., "Added Deduction", "Updated Employee"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    details = db.Column(db.Text, nullable=False)  # JSON or text description of the change       



