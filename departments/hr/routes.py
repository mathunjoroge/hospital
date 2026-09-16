import csv
import io
import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from io import StringIO

from flask import (
    flash,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask_login import current_user, login_required
from flask_mail import Mail, Message
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.exc import SQLAlchemyError

from departments.forms import (
    AddAllowanceForm,
    AddDeductionForm,
    LeaveRequestForm,
    UpdateProfileForm,
)
from departments.models.hr import (
    Allowance,
    AuditLog,
    Deduction,
    Employee,
    Leave,
    Payroll,
    Rota,
)
from departments.models.user import User
from departments.rbac import get_effective_role, roles_required
from extensions import db

from . import bp  # Import the blueprint

logger = logging.getLogger(__name__)


def _current_employee():
    """
    Resolve the Employee record linked to the logged-in User, or None.

    Replaces the previous pattern of using current_user.id as though it were
    an Employee primary key — Employee and User are separate tables with
    independent ID sequences, so that comparison was only ever right by
    coincidence. None means either the account has no linked employee record
    (e.g. it predates Employee.user_id, or is an admin/system account) or the
    caller isn't authenticated; callers must handle that case explicitly
    rather than letting a stray employee_id=None query silently match nothing
    or the wrong row.
    """
    if not current_user.is_authenticated:
        return None
    return Employee.query.filter_by(user_id=current_user.id).first()


@bp.route("/", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def index():
    """HR dashboard displaying key metrics and recent changes."""

    try:
        # Fetch active and inactive employees
        active_employees = Employee.query.filter_by(is_active=True).all()
        inactive_employees = Employee.query.filter_by(is_active=False).all()

        # Count employees
        active_employees_count = len(active_employees)
        inactive_employees_count = len(inactive_employees)
        total_employees_count = active_employees_count + inactive_employees_count

        # Fetch recent changes (e.g., log entries)
        recent_changes = [
            {
                "description": "Employee E0001 marked as inactive.",
                "date": datetime.now(timezone.utc) - timedelta(days=1),
            },
            {
                "description": "New employee E0002 added to the system.",
                "date": datetime.now(timezone.utc) - timedelta(days=2),
            },
        ]  # Replace with actual query logic

        return render_template(
            "hr/index.html",
            active_employees_count=active_employees_count,
            inactive_employees_count=inactive_employees_count,
            total_employees_count=total_employees_count,
            recent_changes=recent_changes,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.index failed: %s", e, exc_info=True)
        return redirect(url_for("home"))


@bp.route("/employee_list", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def employee_list():
    """Displays the list of all employees."""

    try:
        # Fetch all employees
        employees = Employee.query.order_by(Employee.date_hired.desc()).all()
        return render_template("hr/employee_list.html", employees=employees)

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.employee_list failed: %s", e, exc_info=True)
        return redirect(url_for("home"))


@bp.route("/new_employee", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def new_employee():
    """
    Registers a new employee.

    Allowances and deductions are NOT captured per-employee here: Allowance is
    keyed by job_group and Deduction applies platform-wide (see
    generate_payroll(), which pulls both automatically), so there is nothing
    to select at creation time. This route previously tried to assign both
    per-employee anyway via `new_employee.allowances.extend(...)` and a
    per-employee Deduction(employee_id=..., type=...) — neither the
    relationship nor those columns exist on the real models, so every
    submission raised AttributeError before a row was ever written, in
    addition to employee_id generation being undefined entirely.
    """

    try:
        if request.method == "POST":
            # Extract form data
            name = request.form.get("name")
            role = request.form.get("role")
            department = request.form.get("department")
            job_group = request.form.get("job_group")
            basic_salary_raw = (request.form.get("basic_salary") or "").strip()

            # Validate input
            if not all([name, role, department, job_group]):
                raise ValueError("All fields are required!")

            basic_salary = None
            if basic_salary_raw:
                try:
                    basic_salary = float(basic_salary_raw)
                except ValueError as exc:
                    raise ValueError("Basic salary must be a number.") from exc
                if basic_salary < 0:
                    raise ValueError("Basic salary cannot be negative.")

            # Generate a unique employee ID
            employee_id = Employee.generate_employee_id()

            # Create a new employee entry
            new_employee = Employee(
                employee_id=employee_id,
                name=name,
                role=role,
                department=department,
                job_group=job_group,
                basic_salary=basic_salary,
            )
            db.session.add(new_employee)
            db.session.commit()
            flash(f"Employee {name} added successfully!", "success")
            return redirect(url_for("hr.employee_list"))

        # GET: show the current global allowance/deduction rules for
        # reference only — informational, not a per-employee selection.
        allowances = Allowance.query.all()
        deductions = Deduction.query.all()

        return render_template(
            "hr/new_employee.html", allowances=allowances, deductions=deductions
        )

    except (SQLAlchemyError, ValueError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Something went wrong. Please try again.", "error")
        logger.error("hr.new_employee failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.index") if request.method == "GET" else url_for("hr.new_employee"))


@bp.route("/update_employee/<int:employee_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def update_employee(employee_id):
    """Updates an existing employee."""

    try:
        # Fetch the employee by ID
        employee = Employee.query.get_or_404(employee_id)

        if request.method == "POST":
            # Extract form data
            name = request.form.get("name")
            role = request.form.get("role")
            department = request.form.get("department")
            is_active = request.form.get("is_active") == "on"  # Checkbox handling
            basic_salary_raw = (request.form.get("basic_salary") or "").strip()
            linked_username = (request.form.get("linked_username") or "").strip()

            # Validate input
            if not all([name, role, department]):
                raise ValueError("All fields are required!")

            basic_salary = employee.basic_salary
            if basic_salary_raw:
                try:
                    basic_salary = float(basic_salary_raw)
                except ValueError as exc:
                    raise ValueError("Basic salary must be a number.") from exc
                if basic_salary < 0:
                    raise ValueError("Basic salary cannot be negative.")

            if linked_username:
                linked_user = User.query.filter_by(username=linked_username).first()
                if not linked_user:
                    raise ValueError(f"No login account found for username '{linked_username}'.")
                existing_link = Employee.query.filter(
                    Employee.user_id == linked_user.id, Employee.id != employee.id
                ).first()
                if existing_link:
                    raise ValueError(
                        f"That login account is already linked to {existing_link.name}."
                    )
                employee.user_id = linked_user.id
            else:
                employee.user_id = None

            # Update employee details
            employee.name = name
            employee.role = role
            employee.department = department
            employee.is_active = is_active
            employee.basic_salary = basic_salary
            employee.updated_by = current_user.id

            db.session.commit()
            flash(f"Employee {name} updated successfully!", "success")
            return redirect(url_for("hr.employee_list"))

        return render_template("hr/update_employee.html", employee=employee)

    except (SQLAlchemyError, ValueError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Something went wrong. Please try again.", "error")
        logger.error("hr.update_employee failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(
            url_for("hr.update_employee", employee_id=employee_id)
            if request.method == "POST"
            else url_for("hr.employee_list")
        )


@bp.route("/delete_employee/<int:employee_id>", methods=["POST"])
@login_required
@roles_required("hr", "admin")
def delete_employee(employee_id):
    """Deletes an employee from the system."""

    try:
        # Fetch the employee by ID
        employee = Employee.query.get_or_404(employee_id)

        # Delete the employee
        db.session.delete(employee)
        db.session.commit()

        flash(f"Employee {employee.name} deleted successfully!", "success")
        return redirect(url_for("hr.employee_list"))

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.delete_employee failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.employee_list"))


@bp.route("/rota_management", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def rota_management():
    """Manage employee shifts and schedules (rota)."""

    try:
        # Fetch all active employees
        employees = Employee.query.filter_by(is_active=True).all()

        # Fetch all rotas
        rotas = Rota.query.order_by(Rota.week_range.desc()).all()

        # Prepare rota data for rendering
        rota_data = defaultdict(lambda: defaultdict(list))
        for rota in rotas:
            if rota.shift_8_5:
                rota_data[rota.week_range]["morning"] = [
                    name.strip() for name in rota.shift_8_5.split(",")
                ]
            if rota.shift_5_8:
                rota_data[rota.week_range]["evening"] = [rota.shift_5_8.strip()]
            if rota.shift_8_8:
                rota_data[rota.week_range]["night"] = [rota.shift_8_8.strip()]

        if request.method == "POST":
            # Extract form data
            week_range = request.form.get("week_range")  # Selected week range
            if not week_range:
                raise ValueError("Week range is required!")

            # Automatically allocate shifts
            start_date, end_date = week_range.split(" - ")
            start_date = datetime.strptime(start_date.strip(), "%d/%m/%Y").date()  # noqa: DTZ007
            end_date = datetime.strptime(end_date.strip(), "%d/%m/%Y").date()  # noqa: DTZ007

            # Ensure the week range is valid
            if (end_date - start_date).days != 6:
                raise ValueError("Invalid week range! Please specify a 7-day period.")

            # Allocate shifts
            morning_shifts = []
            evening_shifts = []
            night_shifts = []

            # Randomly assign shifts while respecting constraints
            for emp in employees:
                shift_options = ["morning", "evening", "night"]
                assigned_shift = random.choice(shift_options)

                # Prevent consecutive night/evening shifts
                recent_shifts = Rota.query.filter(
                    Rota.week_range >= (start_date - timedelta(days=7)),
                    Rota.week_range <= (start_date + timedelta(days=7)),
                ).all()

                recent_morning = any(
                    emp.name in r.shift_8_5.split(",") if r.shift_8_5 else False
                    for r in recent_shifts
                )
                recent_evening = any(
                    emp.name == r.shift_5_8 if r.shift_5_8 else False
                    for r in recent_shifts
                )
                recent_night = any(
                    emp.name == r.shift_8_8 if r.shift_8_8 else False
                    for r in recent_shifts
                )

                # Apply constraints
                if assigned_shift == "morning" and not recent_morning:
                    morning_shifts.append(emp.name)
                elif assigned_shift == "evening" and not recent_evening:
                    evening_shifts.append(emp.name)
                elif assigned_shift == "night" and not recent_night:
                    night_shifts.append(emp.name)

            # Ensure at least one employee per shift
            if not morning_shifts:
                morning_shifts.append(random.choice([emp.name for emp in employees]))
            if not evening_shifts:
                evening_shifts.append(random.choice([emp.name for emp in employees]))
            if not night_shifts:
                night_shifts.append(random.choice([emp.name for emp in employees]))

            # Save the rota to the database
            rota = next((r for r in rotas if r.week_range == week_range), None)
            if not rota:
                rota = Rota(week_range=week_range)
                db.session.add(rota)

            rota.shift_8_5 = ",".join(morning_shifts)
            rota.shift_5_8 = evening_shifts[0] if evening_shifts else None
            rota.shift_8_8 = night_shifts[0] if night_shifts else None

            db.session.commit()
            flash(f"Rota for {week_range} generated successfully!", "success")
            return redirect(url_for("hr.rota_management"))

        return render_template(
            "hr/rota_management.html",
            employees=employees,
            rotas=rotas,
            rota_data=dict(rota_data),  # Convert defaultdict to dict for Jinja2
        )

    except (SQLAlchemyError, ValueError) as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.rota_management failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.index"))


@bp.route("/department_reports", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def department_reports():
    """Generate department-wise employee distribution reports."""

    try:
        # Fetch all active employees grouped by department and role
        filters = {}
        role_filter = request.args.get(
            "role"
        )  # Optional role filter from query parameters
        if role_filter:
            filters["role"] = role_filter

        employees_by_department = Employee.query.filter_by(
            is_active=True, **filters
        ).all()

        # Group employees by department
        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]["total"] += 1
            department_data[emp.department][emp.role] += 1

        # Convert defaultdict to dict for Jinja2 rendering
        department_data = dict(department_data)

        # Debugging output
        logger.debug("Department employee distribution: %s", department_data)

        return render_template(
            "hr/department_reports.html",
            department_data=department_data,
            role_filter=role_filter,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.department_reports failed: %s", e, exc_info=True)
        return redirect(url_for("hr.index"))


@bp.route("/export_department_reports", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def export_department_reports():
    """Export department-wise employee distribution reports to CSV."""

    try:
        # Fetch all active employees grouped by department and role
        employees_by_department = Employee.query.filter_by(is_active=True).all()

        # Prepare data for CSV
        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]["total"] += 1
            department_data[emp.department][emp.role] += 1

        # Convert data to list of rows
        csv_data = [
            [
                "Department",
                "Total",
                "Records",
                "Nursing",
                "Pharmacy",
                "Medicine",
                "Laboratory",
                "Imaging",
                "Mortuary",
                "HR",
                "Stores",
                "Admin",
            ]
        ]
        for dept, counts in department_data.items():
            row = [
                dept,
                counts["total"],
                counts.get("records", 0),
                counts.get("nursing", 0),
                counts.get("pharmacy", 0),
                counts.get("medicine", 0),
                counts.get("laboratory", 0),
                counts.get("imaging", 0),
                counts.get("mortuary", 0),
                counts.get("hr", 0),
                counts.get("stores", 0),
                counts.get("admin", 0),
            ]
            csv_data.append(row)

        # Generate CSV response
        si = StringIO()
        writer = csv.writer(si)
        writer.writerows(csv_data)
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = (
            "attachment; filename=department_reports.csv"
        )
        output.headers["Content-type"] = "text/csv"
        return output

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.export_department_reports failed: %s", e, exc_info=True)
        return redirect(url_for("hr.department_reports"))


@bp.route("/payroll")
@login_required
@roles_required("hr", "admin")
def payroll_dashboard():
    """Display payroll dashboard with search and filter options."""

    search_query = request.args.get("search", "")
    month_filter = request.args.get("month", "")
    department_filter = request.args.get("department", "")

    query = Payroll.query.join(Employee)
    if search_query:
        query = query.filter(Employee.name.contains(search_query))
    if month_filter:
        query = query.filter(Payroll.month == month_filter)
    if department_filter:
        query = query.filter(Employee.department == department_filter)

    payrolls = query.all()
    return render_template("hr/dashboard.html", payrolls=payrolls)


@bp.route("/generate_payroll/<month>")
@login_required
@roles_required("hr", "admin")
def generate_payroll(month):
    """Generate payroll for a specific month."""

    employees = Employee.query.filter_by(is_active=True).all()
    skipped = []
    generated = 0
    for employee in employees:
        if employee.basic_salary is None:
            skipped.append(employee.name)
            continue

        # Calculate gross pay (basic salary + allowances)
        allowances = Allowance.query.filter_by(job_group=employee.job_group).all()
        total_allowances = sum(allowance.value for allowance in allowances)
        gross_pay = employee.basic_salary + total_allowances

        # Calculate deductions (PAYE, NHIF, NSSF, etc.)
        deductions = Deduction.query.all()
        total_deductions = 0
        for deduction in deductions:
            if deduction.is_percentage:
                total_deductions += gross_pay * (deduction.value / 100)
            else:
                total_deductions += deduction.value

        # Calculate net pay
        net_pay = gross_pay - total_deductions

        # Create payroll record
        payroll = Payroll(
            employee_id=employee.id,
            month=month,
            gross_pay=gross_pay,
            total_deductions=total_deductions,
            net_pay=net_pay,
        )
        db.session.add(payroll)
        generated += 1
    db.session.commit()

    if generated:
        flash(f"Payroll generated for {generated} employee(s).", "success")
    if skipped:
        flash(
            "Skipped (no basic salary on file): " + ", ".join(skipped),
            "warning",
        )
    if not generated and not skipped:
        flash("No active employees to generate payroll for.", "info")
    return redirect(url_for("hr.payroll_dashboard"))


@bp.route("/employee_payroll/<int:employee_id>")
@login_required
@roles_required("hr", "admin")
def employee_payroll(employee_id):
    """Display payroll details for a specific employee."""

    employee = Employee.query.get_or_404(employee_id)
    payrolls = Payroll.query.filter_by(employee_id=employee.id).all()
    return render_template(
        "hr/employee_payroll.html", employee=employee, payrolls=payrolls
    )


@bp.route("/add_deduction", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def add_deduction():
    """Add a new deduction."""

    form = AddDeductionForm()
    if form.validate_on_submit():
        deduction = Deduction(
            name=form.name.data,
            value=form.value.data,
            is_percentage=form.is_percentage.data,
        )
        db.session.add(deduction)
        db.session.commit()
        flash("Deduction added successfully!", "success")
        return redirect(url_for("hr.payroll_dashboard"))
    return render_template("hr/add_deduction.html", form=form)


@bp.route("/add_allowance", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def add_allowance():
    """Add a new allowance."""

    form = AddAllowanceForm()
    if form.validate_on_submit():
        allowance = Allowance(
            job_group=form.job_group.data, name=form.name.data, value=form.value.data
        )
        db.session.add(allowance)
        db.session.commit()
        flash("Allowance added successfully!", "success")
        return redirect(url_for("hr.payroll_dashboard"))
    return render_template("hr/add_allowance.html", form=form)


@bp.route("/leave_request", methods=["GET", "POST"])
@login_required
def leave_request():
    """Submit a leave request for the current employee."""
    # Allow all authenticated employees to submit leave requests
    employee = _current_employee()
    if not employee:
        flash(
            "Your account is not linked to an employee record, so a leave "
            "request cannot be filed. Ask HR to link your account.",
            "error",
        )
        return redirect(url_for("home"))

    form = LeaveRequestForm()
    if form.validate_on_submit():
        leave = Leave(
            employee_id=employee.id,
            start_date=form.start_date.data,
            end_date=form.end_date.data,
            type=form.type.data,
        )
        db.session.add(leave)
        db.session.commit()
        flash("Leave request submitted successfully!", "success")
        return redirect(url_for("hr.index"))
    return render_template("hr/leave_request.html", form=form)


@bp.route("/reports")
@login_required
@roles_required("hr", "admin")
def reports():
    """Generate payroll reports with filters."""

    month = request.args.get("month")
    employee_id = request.args.get("employee_id")
    department = request.args.get("department")

    query = Payroll.query.join(Employee)
    if month:
        query = query.filter(Payroll.month == month)
    if employee_id:
        query = query.filter(Payroll.employee_id == employee_id)
    if department:
        query = query.filter(Employee.department == department)

    payrolls = query.all()
    return render_template("hr/reports.html", payrolls=payrolls)


@bp.route("/audit_logs")
@login_required
@roles_required("hr", "admin")
def audit_logs():
    """Display audit logs."""

    logs = AuditLog.query.all()
    return render_template("hr/audit_logs.html", logs=logs)


@bp.route("/employee_profile/<int:employee_id>")
@login_required
@roles_required("hr", "admin")
def employee_profile(employee_id):
    """Display employee profile."""

    employee = Employee.query.get_or_404(employee_id)
    return render_template("hr/employee_profile.html", employee=employee)


@bp.route("/update_profile/<int:employee_id>", methods=["POST"])
@login_required
@roles_required("hr", "admin")
def update_profile(employee_id):
    """Update employee profile by HR/admin."""

    employee = Employee.query.get_or_404(employee_id)
    employee.name = request.form.get("name")
    employee.department = request.form.get("department")
    employee.job_group = request.form.get("job_group")
    db.session.commit()
    flash("Profile updated successfully!", "success")
    return redirect(url_for("hr.employee_profile", employee_id=employee.id))


@bp.route("/leave_management")
@login_required
@roles_required("hr", "admin")
def leave_management():
    """Manage leave requests."""

    leaves = db.session.query(Leave, Employee).join(Employee).all()
    return render_template("hr/leave_management.html", leaves=leaves)


@bp.route("/reject_leave/<int:leave_id>")
@login_required
@roles_required("hr", "admin")
def reject_leave(leave_id):
    """Reject a leave request."""

    leave = Leave.query.get_or_404(leave_id)
    leave.status = "Rejected"
    db.session.commit()
    flash("Leave request rejected successfully!", "danger")
    return redirect(url_for("hr.leave_management"))


@bp.route("/update_employee_profile", methods=["GET", "POST"])
@login_required
def update_employee_profile():
    """Update profile for the logged-in employee."""
    # Allow all authenticated employees to update their own profile
    employee = _current_employee()
    if not employee:
        flash(
            "Your account is not linked to an employee record. Ask HR to "
            "link your account before updating your profile.",
            "error",
        )
        return redirect(url_for("home"))

    form = UpdateProfileForm()

    if form.validate_on_submit():
        employee.email = form.email.data
        employee.phone = form.phone.data
        employee.bank_name = form.bank_name.data
        employee.bank_account = form.bank_account.data
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("hr.employee_profile", employee_id=employee.id))

    # Pre-fill the form with existing data
    form.email.data = employee.email
    form.phone.data = employee.phone
    form.bank_name.data = employee.bank_name
    form.bank_account.data = employee.bank_account

    return render_template("hr/employee_profile.html", form=form, employee=employee)


@bp.route("/payslips")
@login_required
def employee_payslips():
    """Display payslips for the logged-in employee."""
    # Allow all authenticated employees to view their own payslips
    employee = _current_employee()
    payrolls = Payroll.query.filter_by(employee_id=employee.id).all() if employee else []
    return render_template("hr/employee_payslips.html", payrolls=payrolls)


@bp.route("/export_payroll_pdf")
@login_required
@roles_required("hr", "admin")
def export_payroll_pdf():
    """Export payroll report as PDF."""

    payrolls = Payroll.query.join(Employee).all()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Payroll Report")
    y = 730
    for payroll in payrolls:
        p.drawString(
            100, y, f"{payroll.employee.name} - {payroll.month}: ${payroll.net_pay}"
        )
        y -= 20
    p.showPage()
    p.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="payroll_report.pdf",
        mimetype="application/pdf",
    )


@bp.route("/export_payroll_excel")
@login_required
@roles_required("hr", "admin")
def export_payroll_excel():
    """Export payroll report as CSV (Excel-compatible)."""

    payrolls = Payroll.query.join(Employee).all()

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(["Employee", "Month", "Gross Pay", "Deductions", "Net Pay"])
    for payroll in payrolls:
        writer.writerow(
            [
                payroll.employee.name,
                payroll.month,
                payroll.gross_pay,
                payroll.total_deductions,
                payroll.net_pay,
            ]
        )

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=payroll_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output


def send_email(subject, recipient, body):
    """Send an email notification."""
    msg = Message(subject, recipients=[recipient])
    msg.body = body
    Mail.send(msg)


@bp.route("/approve_leave/<int:leave_id>")
@login_required
@roles_required("hr", "admin")
def approve_leave(leave_id):
    """Approve a leave request."""

    leave = Leave.query.get_or_404(leave_id)
    leave.status = "Approved"
    db.session.commit()

    employee = db.session.get(Employee, leave.employee_id)
    if employee:
        send_email(
            subject="Leave Request Approved",
            recipient=employee.email,
            body=f"Your leave request from {leave.start_date} to {leave.end_date} has been approved.",
        )
    else:
        flash("Employee not found for this leave request!", "warning")

    flash("Leave request approved successfully!", "success")
    return redirect(url_for("hr.leave_management"))


@bp.route("/process_payroll")
@login_required
@roles_required("hr", "admin")
def process_payroll():
    """Process payroll and send notifications."""

    # Process payroll logic here (assumed to be implemented)
    employees = Employee.query.all()
    for employee in employees:
        if employee.payrolls:  # Check if payroll exists
            send_email(
                subject="Payroll Processed",
                recipient=employee.email,
                body=f"Your payroll for the month has been processed. Net Pay: ${employee.payrolls[-1].net_pay}",
            )

    flash("Payroll processed successfully!", "success")
    return redirect(url_for("hr.payroll_dashboard"))


@bp.route("/view_payslip/<int:payroll_id>")
@login_required
def view_payslip(payroll_id):
    """View a specific payslip."""
    # Allow employees to view their own payslips, HR/admins to view all
    payroll = Payroll.query.get_or_404(payroll_id)
    employee = _current_employee()
    if get_effective_role() not in ["hr", "admin"] and (
        not employee or payroll.employee_id != employee.id
    ):
        flash("Unauthorized access. You can only view your own payslips.", "error")
        return redirect(url_for("home"))
    return render_template("hr/payslip.html", payroll=payroll)


@bp.route("/download_payslip/<int:payroll_id>")
@login_required
def download_payslip(payroll_id):
    """Download a specific payslip as PDF."""
    # Allow employees to download their own payslips, HR/admins to download all
    payroll = Payroll.query.get_or_404(payroll_id)
    employee = _current_employee()
    if get_effective_role() not in ["hr", "admin"] and (
        not employee or payroll.employee_id != employee.id
    ):
        flash("Unauthorized access. You can only download your own payslips.", "error")
        return redirect(url_for("home"))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    logo_path = "static/images/logo.png"  # Adjust the path as necessary
    logo = Image(logo_path, width=1.5 * inch, height=1 * inch)
    elements.append(logo)
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Payslip", styles["Title"]))
    elements.append(Paragraph(f"{payroll.month}", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    employee_details = [
        ["Employee Name:", payroll.employee.name],
        ["Employee ID:", payroll.employee.employee_id],
        ["Department:", payroll.employee.department],
        ["Job Group:", payroll.employee.job_group],
        ["Date Hired:", payroll.employee.date_hired.strftime("%Y-%m-%d")],
        ["Status:", "Active" if payroll.employee.is_active else "Inactive"],
    ]
    employee_table = Table(employee_details, colWidths=[2 * inch, 4 * inch])
    employee_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(employee_table)
    elements.append(Spacer(1, 12))

    payroll_details = [
        ["Gross Pay:", f"${payroll.gross_pay}"],
        ["Total Deductions:", f"${payroll.total_deductions}"],
        ["Net Pay:", f"${payroll.net_pay}"],
        ["Payment Date:", payroll.month],
    ]
    payroll_table = Table(payroll_details, colWidths=[2 * inch, 4 * inch])
    payroll_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(payroll_table)
    elements.append(Spacer(1, 12))

    elements.append(
        Paragraph(
            "This is an official payslip generated by Your Company Name. For any discrepancies, please contact HR.",
            styles["BodyText"],
        )
    )
    elements.append(Spacer(1, 12))

    doc.build(elements)

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"payslip_{payroll.employee.name}_{payroll.month}.pdf",
        mimetype="application/pdf",
    )
