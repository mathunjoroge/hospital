from datetime import datetime, timezone

from extensions import db


# Rota Table
class Rota(db.Model):
    """Represents an employee rota."""

    __tablename__ = "rotas"

    id = db.Column(db.Integer, primary_key=True)
    week_range = db.Column(
        db.String(20), nullable=False
    )  # Week range (e.g., "01/03/2023 - 07/03/2023")
    shift_8_5 = db.Column(
        db.Text, nullable=True
    )  # Morning shift (comma-separated names)
    shift_5_8 = db.Column(db.String(100), nullable=True)  # Evening shift
    shift_8_8 = db.Column(db.String(100), nullable=True)  # Night shift

    def __repr__(self):
        return f"<Rota {self.week_range}>"


# Employee Table
class Employee(db.Model):
    """Represents an employee in the HR department."""

    __tablename__ = "employees"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(
        db.String(20), unique=True, nullable=False
    )  # Unique employee ID (e.g., E0001)
    name = db.Column(db.String(100), nullable=False)  # Employee's full name
    role = db.Column(
        db.String(50), nullable=False
    )  # Role/privileges (e.g., records, nursing)
    department = db.Column(db.String(50), nullable=False)  # Department assigned to
    job_group = db.Column(
        db.String(50), nullable=False
    )  # Job group (e.g., "Group A", "Group B")
    date_hired = db.Column(
        db.DateTime, default=datetime.utcnow, nullable=False
    )  # Hire date
    updated_by = db.Column(
        db.Integer, db.ForeignKey("users.id"), nullable=True
    )  # Updated by user
    is_active = db.Column(db.Boolean, default=True, nullable=False)  # Employment status
    email = db.Column(db.String(100), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    bank_name = db.Column(db.String(100), nullable=True)
    bank_account = db.Column(db.String(50), nullable=True)
    # Nullable: an employee not yet assigned a salary is a valid state (new
    # hire pending HR sign-off), not an error. generate_payroll() treats a
    # missing value as "not ready for payroll" and skips the employee rather
    # than crashing the whole run.
    basic_salary = db.Column(db.Numeric(12, 2), nullable=True)

    # Links this HR record to the login account the employee actually uses.
    # Employee and User are separate tables with independent auto-increment
    # sequences — several self-service routes previously compared
    # current_user.id directly against Employee.id, which only "worked" when
    # the two happened to share a number by coincidence. Nullable + unique:
    # not every employee has a login (e.g. contractors on payroll only), and
    # a login must map to at most one employee record.
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), unique=True, nullable=True, index=True
    )
    user = db.relationship("User", foreign_keys=[user_id])

    def __repr__(self):
        return f"<Employee {self.name} - ID: {self.employee_id}>"

    @staticmethod
    def generate_employee_id():
        """
        Generate a unique employee ID (E0001, E0002, ...), matching the
        convention Patient.generate_patient_id() uses for patient_id.

        This was previously called from hr.routes.new_employee() as
        Employee.generate_employee_id() without ever being defined, so every
        attempt to add an employee through the UI raised AttributeError.
        """
        prefix = "E"
        with db.session.no_autoflush:
            last_employee = Employee.query.order_by(Employee.id.desc()).first()
            if last_employee and last_employee.employee_id.startswith(prefix):
                tail = last_employee.employee_id[1:]
                last_number = int(tail) if tail.isdigit() else 0
            else:
                last_number = 0
            return f"{prefix}{last_number + 1:04d}"


class Allowance(db.Model):
    """Represents an allowance for a specific job group."""

    __tablename__ = "allowances"

    id = db.Column(db.Integer, primary_key=True)
    job_group = db.Column(
        db.String(50), nullable=False
    )  # Job group (e.g., "Group A", "Group B")
    name = db.Column(
        db.String(100), nullable=False
    )  # Allowance name (e.g., "Housing Allowance")
    value = db.Column(db.Numeric(12, 2), nullable=False)  # Absolute allowance value

    def __repr__(self):
        return f"<Allowance {self.name} - Job Group: {self.job_group}, Value: {self.value}>"


class Payroll(db.Model):
    """Represents a payroll record for an employee."""

    __tablename__ = "payrolls"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    month = db.Column(
        db.String(20), nullable=False
    )  # Payroll month (e.g., "January 2023")
    gross_pay = db.Column(
        db.Numeric(12, 2), nullable=False
    )  # Total earnings (basic salary + allowances)
    total_deductions = db.Column(db.Numeric(12, 2), nullable=False)  # Total deductions
    net_pay = db.Column(
        db.Numeric(12, 2), nullable=False
    )  # Net pay (gross_pay - total_deductions)

    employee = db.relationship("Employee", backref="payrolls")

    def __repr__(self):
        return f"<Payroll {self.month} - Employee: {self.employee.name}, Net Pay: {self.net_pay}>"


class Deduction(db.Model):
    """Represents a deduction (e.g., PAYE, NHIF, NSSF)."""

    __tablename__ = "deductions"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)  # Deduction name (e.g., "PAYE")
    value = db.Column(db.Numeric(12, 4), nullable=False)  # Deduction value (fixed or percentage; 4dp for PAYE/NHIF rates)
    is_percentage = db.Column(
        db.Boolean, default=False
    )  # Whether the value is a percentage

    def __repr__(self):
        return f"<Deduction {self.name} - Value: {self.value}>"


class Leave(db.Model):
    __tablename__ = "leaves"
    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    type = db.Column(db.String(50), nullable=False)  # e.g., sick leave, vacation
    status = db.Column(
        db.String(20), default="Pending"
    )  # e.g., Pending, Approved, Rejected

    # employee_profile.html renders an employee's leave history via
    # employee.leaves; without this relationship the template raised
    # UndefinedError on every profile visit.
    employee = db.relationship("Employee", backref="leaves")




class AuditLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    action = db.Column(
        db.String(100), nullable=False
    )  # e.g., "Added Deduction", "Updated Employee"
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    details = db.Column(
        db.Text, nullable=False
    )  # JSON or text description of the change


class StaffCredential(db.Model):
    """Staff Professional License & Credential Tracking Model."""

    __tablename__ = "staff_credentials"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=True)
    staff_name = db.Column(db.String(100), nullable=False)  # e.g., 'Dr. Jane Smith'
    credential_type = db.Column(
        db.String(100), nullable=False
    )  # e.g. KMPDC, NCK, PPB, Radiography Board
    credential_number = db.Column(db.String(100), nullable=False)
    issue_date = db.Column(db.Date, nullable=True)
    expiry_date = db.Column(db.Date, nullable=False, index=True)
    status = db.Column(
        db.String(20), default="ACTIVE", nullable=False
    )  # ACTIVE, EXPIRED, RENEWAL_PENDING
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    employee = db.relationship("Employee", backref=db.backref("credentials", lazy=True))

    @property
    def days_until_expiry(self) -> int:
        if not self.expiry_date:
            return 999

        return (self.expiry_date - datetime.now(timezone.utc).date()).days

    def __repr__(self):
        return f"<StaffCredential {self.staff_name} - {self.credential_type} ({self.credential_number})>"
