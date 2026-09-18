from datetime import datetime, timezone

from sqlalchemy import extract

from extensions import db

# Statutory minimum annual leave entitlement (Kenya Employment Act, s.28):
# 21 working days per completed 12 months of service. Used as the default
# for Employee.annual_leave_days so existing rows (and the seed data) get a
# sane value without HR having to configure anything on day one.
DEFAULT_ANNUAL_LEAVE_DAYS = 21


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

    # ── Contract / probation ──────────────────────────────────────────────
    contract_type = db.Column(
        db.String(30), nullable=True, default="permanent"
    )  # permanent | contract | locum | intern
    probation_end_date = db.Column(
        db.Date, nullable=True
    )  # NULL means confirmed/not on probation
    national_id = db.Column(
        db.String(30), nullable=True
    )  # National ID / Passport number
    kra_pin = db.Column(
        db.String(20), nullable=True
    )  # KRA PIN for payroll statutory purposes

    # ── Next of kin / emergency contact ──────────────────────────────────
    nok_name = db.Column(db.String(100), nullable=True)
    nok_phone = db.Column(db.String(20), nullable=True)
    nok_relationship = db.Column(
        db.String(50), nullable=True
    )  # e.g. spouse, parent, sibling

    # Links this HR record to the login account the employee actually uses.
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id"), unique=True, nullable=True, index=True
    )
    user = db.relationship("User", foreign_keys=[user_id])

    # Annual (vacation) leave entitlement in working days.
    annual_leave_days = db.Column(
        db.Integer, nullable=False, default=DEFAULT_ANNUAL_LEAVE_DAYS
    )

    def __repr__(self):
        return f"<Employee {self.name} - ID: {self.employee_id}>"

    def leave_days_used(self, leave_type="vacation", year=None):
        """
        Sum of Approved leave days of `leave_type` for this employee in
        `year` (defaults to the current year). Pending/Rejected requests
        don't count against the balance — only leave HR has actually
        approved does.
        """
        year = year or datetime.now(timezone.utc).year
        approved = Leave.query.filter(
            Leave.employee_id == self.id,
            Leave.type == leave_type,
            Leave.status == "Approved",
            extract("year", Leave.start_date) == year,
        ).all()
        return sum(leave.days for leave in approved)

    def leave_days_remaining(self, year=None):
        """Vacation days left against annual_leave_days for `year`."""
        entitlement = self.annual_leave_days or 0
        return entitlement - self.leave_days_used(leave_type="vacation", year=year)

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
    value = db.Column(
        db.Numeric(12, 4), nullable=False
    )  # Deduction value (fixed or percentage; 4dp for PAYE/NHIF rates)
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
    reason = db.Column(db.Text, nullable=True)  # Optional reason/notes from employee
    approved_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    approved_at = db.Column(db.DateTime, nullable=True)

    # employee_profile.html renders an employee's leave history via
    # employee.leaves; without this relationship the template raised
    # UndefinedError on every profile visit.
    employee = db.relationship("Employee", backref="leaves")

    @property
    def days(self):
        """Inclusive day count (a single-day leave is 1 day, not 0)."""
        if not self.start_date or not self.end_date:
            return 0
        return (self.end_date - self.start_date).days + 1


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

    # Nothing ever populated this table (hr.routes had no write path at
    # all), so hr/audit_logs.html always rendered an empty table and the HR
    # dashboard's "Recent Changes" card was backed by two hardcoded fake
    # entries instead of anything that actually happened. The relationship
    # lets the template show a username instead of a bare numeric user_id.
    user = db.relationship("User", foreign_keys=[user_id])


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


class LeaveBalance(db.Model):
    """
    Annual leave entitlement and usage per employee per year.

    Kenyan Employment Act: permanent employees earn 21 working days annual
    leave per year minimum. Sick leave and other statutory types are
    tracked here so HR can see remaining balances at a glance.
    """

    __tablename__ = "leave_balances"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    year = db.Column(db.Integer, nullable=False)  # e.g. 2026
    leave_type = db.Column(
        db.String(50), nullable=False
    )  # annual | sick | maternity | …
    entitled_days = db.Column(db.Integer, nullable=False, default=21)
    used_days = db.Column(db.Integer, nullable=False, default=0)
    carried_over = db.Column(
        db.Integer, nullable=False, default=0
    )  # days carried from prior year

    employee = db.relationship("Employee", backref="leave_balances")

    __table_args__ = (
        db.UniqueConstraint(
            "employee_id", "year", "leave_type", name="uq_leave_balance"
        ),
    )

    @property
    def remaining_days(self):
        return max(0, self.entitled_days + self.carried_over - self.used_days)

    def __repr__(self):
        return f"<LeaveBalance emp={self.employee_id} year={self.year} type={self.leave_type} remaining={self.remaining_days}>"


class PerformanceReview(db.Model):
    """
    Staff performance appraisal record.

    Supports annual / mid-year / probation-review cycles.
    Score is 1–5 (1=Unsatisfactory, 3=Meets Expectations, 5=Outstanding).
    """

    __tablename__ = "performance_reviews"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    reviewer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    review_period = db.Column(
        db.String(50), nullable=False
    )  # e.g. "2026-H1", "2026-Annual"
    review_type = db.Column(
        db.String(30), nullable=False, default="annual"
    )  # annual | probation | mid-year
    score = db.Column(db.Integer, nullable=False)  # 1–5
    strengths = db.Column(db.Text, nullable=True)
    areas_for_improvement = db.Column(db.Text, nullable=True)
    goals_next_period = db.Column(db.Text, nullable=True)
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    employee = db.relationship("Employee", backref="performance_reviews")
    reviewer = db.relationship("User", foreign_keys=[reviewer_id])

    def __repr__(self):
        return f"<PerformanceReview emp={self.employee_id} period={self.review_period} score={self.score}>"


class TrainingRecord(db.Model):
    """
    Staff training, CPD (Continuing Professional Development), and
    certification completion records.

    Regulatory bodies (KMPDC, NCK, PPB) require CPD points — this table
    provides an auditable log per employee.
    """

    __tablename__ = "training_records"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    title = db.Column(db.String(200), nullable=False)  # Course / workshop title
    provider = db.Column(db.String(200), nullable=True)  # Training institution
    training_type = db.Column(db.String(50), nullable=False, default="cpd")
    # cpd | mandatory | skills | leadership | induction | conference
    date_completed = db.Column(db.Date, nullable=False)
    expiry_date = db.Column(db.Date, nullable=True)  # If certification expires
    cpd_points = db.Column(db.Integer, nullable=True)  # CPD points earned
    certificate_number = db.Column(db.String(100), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    recorded_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    employee = db.relationship("Employee", backref="training_records")
    recorder = db.relationship("User", foreign_keys=[recorded_by])

    def __repr__(self):
        return f"<TrainingRecord emp={self.employee_id} title={self.title!r}>"


class DisciplinaryRecord(db.Model):
    """
    Staff disciplinary action log.

    Kenyan Employment Act s.41 requires a fair hearing and written record
    before any summary dismissal. This table provides the auditable trail
    HR needs.
    """

    __tablename__ = "disciplinary_records"

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey("employees.id"), nullable=False)
    incident_date = db.Column(db.Date, nullable=False)
    incident_type = db.Column(db.String(50), nullable=False)
    # verbal_warning | written_warning | final_warning | suspension | dismissal | other
    description = db.Column(db.Text, nullable=False)  # What happened
    action_taken = db.Column(db.Text, nullable=False)  # What HR/management did
    outcome = db.Column(
        db.String(50), nullable=True
    )  # resolved | appeal_pending | dismissed
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)  # False = expunged

    employee = db.relationship("Employee", backref="disciplinary_records")
    reviewer = db.relationship("User", foreign_keys=[reviewed_by])

    def __repr__(self):
        return f"<DisciplinaryRecord emp={self.employee_id} type={self.incident_type} date={self.incident_date}>"
