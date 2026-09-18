import csv
import io
import logging
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from io import StringIO

from flask import (
    current_app,
    flash,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from flask_mail import Message
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
    DisciplinaryRecordForm,
    LeaveRequestForm,
    PerformanceReviewForm,
    TrainingRecordForm,
    UpdateProfileForm,
)
from departments.models.hr import (
    Allowance,
    AuditLog,
    Deduction,
    DisciplinaryRecord,
    Employee,
    Leave,
    LeaveBalance,
    Payroll,
    PerformanceReview,
    Rota,
    StaffCredential,
    TrainingRecord,
)
from departments.models.user import User
from departments.rbac import get_effective_role, roles_required
from extensions import db

from . import bp  # Import the blueprint

logger = logging.getLogger(__name__)


# ── Internal helpers ───────────────────────────────────────────────────────

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


def _log_hr_action(action, details):
    """
    Record an HR audit trail entry, best-effort.

    Nothing previously wrote to AuditLog at all: hr/audit_logs.html always
    rendered an empty table, and the HR dashboard's "Recent Changes" card
    was backed by two hardcoded fake entries (see the old index() comment
    "Replace with actual query logic"). This is called from every HR route
    that actually changes something, mirroring the pattern send_email()
    already uses — a logging failure must never roll back or mask the
    change it's describing.
    """
    try:
        db.session.add(
            AuditLog(
                user_id=current_user.id if current_user.is_authenticated else None,
                action=action,
                details=details,
            )
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        logger.exception("hr._log_hr_action failed: action=%r", action)


_write_audit = _log_hr_action


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

        # Recent changes, from the real HR audit trail. This used to be two
        # hardcoded entries ("Employee E0001 marked as inactive...") that
        # never changed no matter what HR actually did, because nothing
        # wrote to AuditLog — see _log_hr_action(), now called from every
        # mutating HR route.
        recent_logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(10).all()
        recent_changes = [
            {
                "description": f"{log.action} — {log.details}",
                "date": log.timestamp,
            }
            for log in recent_logs
        ]

        today = datetime.now(timezone.utc).date()
        expiring_credentials = StaffCredential.query.filter(
            StaffCredential.expiry_date <= today + timedelta(days=30),
            StaffCredential.expiry_date >= today,
            StaffCredential.status == "ACTIVE",
        ).count()

        pending_leaves = Leave.query.filter_by(status="Pending").count()

        return render_template(
            "hr/index.html",
            active_employees_count=active_employees_count,
            inactive_employees_count=inactive_employees_count,
            total_employees_count=total_employees_count,
            recent_changes=recent_changes,
            expiring_credentials=expiring_credentials,
            pending_leaves=pending_leaves,
        )

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.index failed: %s", e, exc_info=True)
        return redirect(url_for("home"))


@bp.route("/employee_list", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def employee_list():
    """
    Displays the list of employees, filterable by status/department and
    searchable by name or employee ID.

    The HR dashboard has always linked here with ?status=active and
    ?status=inactive (see hr/index.html's "Active Employees" / "Inactive
    Employees" cards), but this route ignored request.args entirely and
    always returned every employee — so both cards silently opened the
    exact same unfiltered list instead of what they promised.
    """

    try:
        search = (request.args.get("search") or "").strip()
        status = (request.args.get("status") or "").strip().lower()
        department_filter = (request.args.get("department") or "").strip()

        query = Employee.query
        if status == "active":
            query = query.filter_by(is_active=True)
        elif status == "inactive":
            query = query.filter_by(is_active=False)

        if department_filter:
            query = query.filter(Employee.department == department_filter)

        if search:
            like = f"%{search}%"
            query = query.filter(
                db.or_(Employee.name.ilike(like), Employee.employee_id.ilike(like))
            )

        employees = query.order_by(Employee.date_hired.desc()).all()

        # Distinct departments for the filter dropdown, independent of the
        # current filters so switching department doesn't shrink the list
        # of choices offered.
        departments = [
            row[0]
            for row in db.session.query(Employee.department)
            .distinct()
            .order_by(Employee.department)
            .all()
            if row[0]
        ]

        return render_template(
            "hr/employee_list.html",
            employees=employees,
            departments=departments,
            search=search,
            status=status,
            department_filter=department_filter,
        )

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
    to select at creation time.
    """

    try:
        if request.method == "POST":
            name = request.form.get("name")
            role = request.form.get("role")
            department = request.form.get("department")
            job_group = request.form.get("job_group")
            basic_salary_raw = (request.form.get("basic_salary") or "").strip()

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
            _log_hr_action(
                "Employee Created",
                f"{employee_id} ({name}) added to {department}, job group {job_group}.",
            )
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
        employee = Employee.query.get_or_404(employee_id)

        if request.method == "POST":
            name = request.form.get("name")
            role = request.form.get("role")
            department = request.form.get("department")
            is_active = request.form.get("is_active") == "on"
            basic_salary_raw = (request.form.get("basic_salary") or "").strip()
            linked_username = (request.form.get("linked_username") or "").strip()

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
            _log_hr_action(
                "Employee Updated",
                f"{employee.employee_id} ({name}) updated by {current_user.username}.",
            )
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
    """Deletes an employee — or deactivates if payroll/leave history exists."""

    try:
        employee = Employee.query.get_or_404(employee_id)

        has_history = (
            Payroll.query.filter_by(employee_id=employee.id).first() is not None
            or Leave.query.filter_by(employee_id=employee.id).first() is not None
        )
        if has_history:
            employee.is_active = False
            employee.updated_by = current_user.id
            db.session.commit()
            _log_hr_action(
                "Employee Deactivated",
                f"{employee.employee_id} ({employee.name}) deactivated "
                "(payroll/leave history retained).",
            )
            flash(
                f"{employee.name} has payroll/leave history, so the record was "
                "deactivated instead of deleted.",
                "warning",
            )
            return redirect(url_for("hr.employee_list"))

        employee_id_label, employee_name = employee.employee_id, employee.name
        db.session.delete(employee)
        db.session.commit()
        _log_hr_action(
            "Employee Deleted",
            f"{employee_id_label} ({employee_name}) permanently deleted "
            f"by {current_user.username} (no payroll/leave history).",
        )

        flash(f"Employee {employee.name} deleted successfully!", "success")
        return redirect(url_for("hr.employee_list"))

    except SQLAlchemyError as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.delete_employee failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.employee_list"))


# ── Rota ───────────────────────────────────────────────────────────────────

@bp.route("/rota_management", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def rota_management():
    """Manage employee shifts and schedules (rota)."""

    try:
        employees = Employee.query.filter_by(is_active=True).all()
        rotas = Rota.query.order_by(Rota.week_range.desc()).all()

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
            week_range = request.form.get("week_range")
            if not week_range:
                raise ValueError("Week range is required!")

            start_date, end_date = week_range.split(" - ")
            start_date = datetime.strptime(start_date.strip(), "%d/%m/%Y").date()  # noqa: DTZ007
            end_date = datetime.strptime(end_date.strip(), "%d/%m/%Y").date()  # noqa: DTZ007

            if (end_date - start_date).days != 6:
                raise ValueError("Invalid week range! Please specify a 7-day period.")

            morning_shifts = []
            evening_shifts = []
            night_shifts = []

            for emp in employees:
                shift_options = ["morning", "evening", "night"]
                assigned_shift = random.choice(shift_options)

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

                if assigned_shift == "morning" and not recent_morning:
                    morning_shifts.append(emp.name)
                elif assigned_shift == "evening" and not recent_evening:
                    evening_shifts.append(emp.name)
                elif assigned_shift == "night" and not recent_night:
                    night_shifts.append(emp.name)

            if not morning_shifts:
                morning_shifts.append(random.choice([emp.name for emp in employees]))
            if not evening_shifts:
                evening_shifts.append(random.choice([emp.name for emp in employees]))
            if not night_shifts:
                night_shifts.append(random.choice([emp.name for emp in employees]))

            rota = next((r for r in rotas if r.week_range == week_range), None)
            if not rota:
                rota = Rota(week_range=week_range)
                db.session.add(rota)

            rota.shift_8_5 = ",".join(morning_shifts)
            rota.shift_5_8 = evening_shifts[0] if evening_shifts else None
            rota.shift_8_8 = night_shifts[0] if night_shifts else None

            db.session.commit()
            _log_hr_action("Rota Generated", f"Rota generated for {week_range}.")
            flash(f"Rota for {week_range} generated successfully!", "success")
            return redirect(url_for("hr.rota_management"))

        return render_template(
            "hr/rota_management.html",
            employees=employees,
            rotas=rotas,
            rota_data=dict(rota_data),
        )

    except (SQLAlchemyError, ValueError) as e:
        flash("Something went wrong. Please try again.", "error")
        logger.error("hr.rota_management failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.index"))


# ── Reports & Exports ──────────────────────────────────────────────────────

@bp.route("/department_reports", methods=["GET"])
@login_required
@roles_required("hr", "admin")
def department_reports():
    """Generate department-wise employee distribution reports."""

    try:
        filters = {}
        role_filter = request.args.get("role")
        if role_filter:
            filters["role"] = role_filter

        employees_by_department = Employee.query.filter_by(
            is_active=True, **filters
        ).all()

        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]["total"] += 1
            department_data[emp.department][emp.role] += 1

        department_data = dict(department_data)

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
        employees_by_department = Employee.query.filter_by(is_active=True).all()

        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]["total"] += 1
            department_data[emp.department][emp.role] += 1

        csv_data = [
            [
                "Department", "Total", "Records", "Nursing", "Pharmacy", "Medicine",
                "Laboratory", "Imaging", "Mortuary", "HR", "Stores", "Admin",
            ]
        ]
        for dept, counts in department_data.items():
            row = [
                dept, counts["total"],
                counts.get("records", 0), counts.get("nursing", 0),
                counts.get("pharmacy", 0), counts.get("medicine", 0),
                counts.get("laboratory", 0), counts.get("imaging", 0),
                counts.get("mortuary", 0), counts.get("hr", 0),
                counts.get("stores", 0), counts.get("admin", 0),
            ]
            csv_data.append(row)

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


# ── Payroll ────────────────────────────────────────────────────────────────

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
    """
    Generate payroll for a specific month.

    Re-running this for a month it's already been run for used to insert a
    second Payroll row per employee — nothing checked for an existing
    record first — silently doubling every reported gross/net figure and
    leaving two payslips per employee for the same month. Existing records
    are now recalculated in place instead of duplicated, so generate_payroll
    is safe to re-run after a late allowance/deduction change.
    """

    employees = Employee.query.filter_by(is_active=True).all()
    skipped = []
    created = 0
    updated = 0
    for employee in employees:
        if employee.basic_salary is None:
            skipped.append(employee.name)
            continue

        allowances = Allowance.query.filter_by(job_group=employee.job_group).all()
        total_allowances = sum(allowance.value for allowance in allowances)
        gross_pay = employee.basic_salary + total_allowances

        deductions = Deduction.query.all()
        total_deductions = 0
        for deduction in deductions:
            if deduction.is_percentage:
                total_deductions += gross_pay * (deduction.value / 100)
            else:
                total_deductions += deduction.value

        # Calculate net pay
        net_pay = gross_pay - total_deductions

        payroll = Payroll.query.filter_by(employee_id=employee.id, month=month).first()
        if payroll:
            payroll.gross_pay = gross_pay
            payroll.total_deductions = total_deductions
            payroll.net_pay = net_pay
            updated += 1
        else:
            payroll = Payroll(
                employee_id=employee.id,
                month=month,
                gross_pay=gross_pay,
                total_deductions=total_deductions,
                net_pay=net_pay,
            )
            db.session.add(payroll)
            created += 1
    db.session.commit()
    _log_hr_action(
        "Payroll Generated",
        f"{month}: {created} created, {updated} recalculated, "
        f"{len(skipped)} skipped (no basic salary).",
    )

    if created:
        flash(f"Payroll generated for {created} employee(s).", "success")
    if updated:
        flash(
            f"Payroll for {updated} employee(s) already existed for {month} "
            "and was recalculated instead of duplicated.",
            "info",
        )
    if skipped:
        flash(
            "Skipped (no basic salary on file): " + ", ".join(skipped),
            "warning",
        )
    if not created and not updated and not skipped:
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
        _log_hr_action(
            "Deduction Added",
            f"{deduction.name}: "
            + (f"{deduction.value}%" if deduction.is_percentage else str(deduction.value))
            + " (applies to all employees).",
        )
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
        _log_hr_action(
            "Allowance Added",
            f"{allowance.name}: {allowance.value} for job group {allowance.job_group}.",
        )
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
        # DateField gives back date objects, but Leave.start_date/end_date
        # are DateTime columns — comparing/storing a bare date next to rows
        # stored as datetime (as approve_leave's own history does) risks a
        # dtype mismatch in the overlap query below on some backends, so
        # normalize to datetime once, here.
        start_date = datetime.combine(form.start_date.data, datetime.min.time())
        end_date = datetime.combine(form.end_date.data, datetime.min.time())
        leave_type = form.type.data
        requested_days = (end_date - start_date).days + 1

        # Two employees, or the same employee twice, could otherwise both
        # have a Pending/Approved leave covering the same days — nothing
        # previously checked this, so approve_leave() would happily approve
        # overlapping time off. Standard interval-overlap test: two ranges
        # overlap unless one ends before the other starts.
        overlapping = Leave.query.filter(
            Leave.employee_id == employee.id,
            Leave.status.in_(["Pending", "Approved"]),
            Leave.start_date <= end_date,
            Leave.end_date >= start_date,
        ).first()
        if overlapping:
            flash(
                f"You already have a {overlapping.status.lower()} "
                f"{overlapping.type} request from "
                f"{overlapping.start_date.date()} to {overlapping.end_date.date()} "
                "that overlaps these dates.",
                "error",
            )
            return render_template("hr/leave_request.html", form=form, employee=employee)

        # Only vacation is capped against the annual entitlement; sick leave
        # (and any other future leave type) is uncapped here by design —
        # HR reviews those on approval instead of the system hard-blocking
        # a genuinely sick employee.
        if leave_type == "vacation":
            remaining = employee.leave_days_remaining()
            if requested_days > remaining:
                flash(
                    f"This request is for {requested_days} day(s) of vacation, "
                    f"but only {remaining} remain this year.",
                    "error",
                )
                return render_template(
                    "hr/leave_request.html", form=form, employee=employee
                )

        leave = Leave(
            employee_id=employee.id,
            start_date=start_date,
            end_date=end_date,
            type=leave_type,
        )
        db.session.add(leave)
        db.session.commit()
        _log_hr_action(
            "Leave Requested",
            f"{employee.employee_id} ({employee.name}) requested {requested_days} "
            f"day(s) {leave_type} leave, {start_date} to {end_date}.",
        )
        flash("Leave request submitted successfully!", "success")
        # hr.index is roles_required("hr", "admin") — redirecting there sent
        # every non-HR employee straight into a 403 right after their request
        # was filed. Their own profile page lists their leave requests, so they
        # land somewhere that shows the result (employee_profile permits the
        # linked employee to view their own record).
        return redirect(url_for("hr.employee_profile", employee_id=employee.id))
    return render_template("hr/leave_request.html", form=form, employee=employee)


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

    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).all()
    return render_template("hr/audit_logs.html", logs=logs)


@bp.route("/employee_profile/<int:employee_id>")
@login_required
def employee_profile(employee_id):
    """
    Display employee profile — HR/admin for anyone, or the linked employee
    for their own record (same ownership check view_payslip uses).

    Previously roles_required("hr", "admin") only, which 403'd the self-service
    flows that land here: update_employee_profile and update_profile redirect
    to this page after a successful save, so every non-HR employee who updated
    their contact/bank details was bounced with 403 *after* the write.
    """
    employee = Employee.query.get_or_404(employee_id)
    if get_effective_role() not in ["hr", "admin"]:
        own_employee = _current_employee()
        if not own_employee or own_employee.id != employee.id:
            flash("Unauthorized access. You can only view your own profile.", "error")
            return redirect(url_for("home"))
    return render_template("hr/employee_profile.html", employee=employee)


@bp.route("/update_profile/<int:employee_id>", methods=["POST"])
@login_required
def update_profile(employee_id):
    """Update employee profile by HR/admin or by the linked employee for their own record."""
    employee = Employee.query.get_or_404(employee_id)
    is_hr = get_effective_role() in ["hr", "admin"]
    current_employee = _current_employee()
    if not is_hr and (not current_employee or current_employee.id != employee.id):
        flash("Unauthorized access. You can only update your own profile.", "error")
        return redirect(url_for("home"))

    try:
        name = (request.form.get("name") or "").strip()
        department = (request.form.get("department") or "").strip()
        job_group = (request.form.get("job_group") or "").strip()
        if not all([name, department, job_group]):
            raise ValueError("All fields are required!")

        employee.name = name
        if is_hr:
            employee.department = department
            employee.job_group = job_group
            employee.updated_by = current_user.id
        db.session.commit()
        flash("Profile updated successfully!", "success")
        return redirect(url_for("hr.employee_profile", employee_id=employee.id))

    except (SQLAlchemyError, ValueError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Something went wrong. Please try again.", "error")
        logger.error("hr.update_profile failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.employee_profile", employee_id=employee_id))


@bp.route("/leave_management")
@login_required
@roles_required("hr", "admin")
def leave_management():
    """Manage leave requests."""

    leaves = db.session.query(Leave, Employee).join(Employee).all()
    return render_template("hr/leave_management.html", leaves=leaves)


@bp.route("/reject_leave/<int:leave_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def reject_leave(leave_id):
    """Reject a leave request."""

    leave = Leave.query.get_or_404(leave_id)
    leave.status = "Rejected"
    db.session.commit()
    employee = db.session.get(Employee, leave.employee_id)
    if employee:
        _log_hr_action(
            "Leave Rejected",
            f"{employee.employee_id} ({employee.name}) {leave.type} leave "
            f"{leave.start_date.date()} to {leave.end_date.date()} rejected "
            f"by {current_user.username}.",
        )
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
    """Export full payroll report as PDF."""

    payrolls = Payroll.query.join(Employee).all()

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Payroll Report")
    y = 730
    for payroll in payrolls:
        p.drawString(
            100, y, f"{payroll.employee.name} - {payroll.month}: KES {payroll.net_pay:,.2f}"
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
    """
    Send an email notification, best-effort.

    The previous implementation called Mail.send(msg) on the flask_mail.Mail
    CLASS — the app-bound instance lives in app.py and was never reachable
    from here — so every call raised TypeError. approve_leave() had already
    committed the status change by then, so HR saw a 500 after the leave was
    approved. It also passed recipients=[None] when the employee had no email
    on file, which raises inside flask_mail itself. Both failure modes are
    non-fatal now: a missing mail extension or recipient logs and moves on,
    never after the DB write has been allowed to fail.
    """
    if not recipient:
        logger.warning(
            "hr.send_email skipped (no recipient on file): subject=%r", subject
        )
        return False
    mail_ext = current_app.extensions.get("mail")
    if mail_ext is None:
        logger.warning(
            "hr.send_email skipped (mail extension not initialised): subject=%r",
            subject,
        )
        return False
    try:
        msg = Message(subject, recipients=[recipient])
        msg.body = body
        mail_ext.send(msg)
        return True
    except Exception:  # noqa: BLE001 — notification must never break the workflow
        logger.exception("hr.send_email failed: subject=%r", subject)
        return False


@bp.route("/approve_leave/<int:leave_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def approve_leave(leave_id):
    """Approve a leave request."""

    leave = Leave.query.get_or_404(leave_id)
    leave.status = "Approved"
    db.session.commit()

    employee = db.session.get(Employee, leave.employee_id)
    if employee:
        _log_hr_action(
            "Leave Approved",
            f"{employee.employee_id} ({employee.name}) {leave.type} leave "
            f"{leave.start_date.date()} to {leave.end_date.date()} approved "
            f"by {current_user.username}.",
        )
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

    employees = Employee.query.all()
    for employee in employees:
        if employee.payrolls:
            send_email(
                subject="Payroll Processed",
                recipient=employee.email,
                body=f"Your payroll for the month has been processed. Net Pay: KES {employee.payrolls[-1].net_pay:,.2f}",
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

    logo_path = "static/images/logo.png"
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
        ["Gross Pay:", f"KES {payroll.gross_pay:,.2f}"],
        ["Total Deductions:", f"KES {payroll.total_deductions:,.2f}"],
        ["Net Pay:", f"KES {payroll.net_pay:,.2f}"],
        ["Payment Month:", payroll.month],
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
            "This is an official payslip. For any discrepancies, please contact HR.",
            styles["BodyText"],
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"payslip_{payroll.employee.name}_{payroll.month}.pdf",
        mimetype="application/pdf",
    )


# ── Leave Balance ──────────────────────────────────────────────────────────

@bp.route("/leave_balance")
@login_required
@roles_required("hr", "admin")
def leave_balance():
    """View and manage leave balances for all employees."""

    year = request.args.get("year", datetime.now(timezone.utc).year, type=int)
    employee_id = request.args.get("employee_id", type=int)

    query = LeaveBalance.query.filter_by(year=year)
    if employee_id:
        query = query.filter_by(employee_id=employee_id)

    balances = query.join(Employee).order_by(Employee.name).all()
    employees = Employee.query.filter_by(is_active=True).order_by(Employee.name).all()

    return render_template(
        "hr/leave_balance.html",
        balances=balances,
        employees=employees,
        selected_year=year,
        selected_employee_id=employee_id,
    )


@bp.route("/leave_balance/set", methods=["POST"])
@login_required
@roles_required("hr", "admin")
def set_leave_balance():
    """Create or update a leave balance record for an employee."""

    try:
        employee_id = request.form.get("employee_id", type=int)
        year = request.form.get("year", type=int)
        leave_type = request.form.get("leave_type")
        entitled_days = request.form.get("entitled_days", type=int)
        carried_over = request.form.get("carried_over", 0, type=int)

        if not all([employee_id, year, leave_type, entitled_days]):
            raise ValueError("All fields are required.")

        Employee.query.get_or_404(employee_id)

        balance = LeaveBalance.query.filter_by(
            employee_id=employee_id, year=year, leave_type=leave_type
        ).first()

        if balance:
            balance.entitled_days = entitled_days
            balance.carried_over = carried_over
        else:
            balance = LeaveBalance(
                employee_id=employee_id,
                year=year,
                leave_type=leave_type,
                entitled_days=entitled_days,
                carried_over=carried_over,
            )
            db.session.add(balance)

        _write_audit("Set Leave Balance", {
            "employee_id": employee_id, "year": year,
            "leave_type": leave_type, "entitled_days": entitled_days,
        })
        db.session.commit()
        flash("Leave balance updated.", "success")

    except (SQLAlchemyError, ValueError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Something went wrong.", "error")
        db.session.rollback()

    return redirect(url_for("hr.leave_balance"))


# ── Staff Credentials (HR view) ────────────────────────────────────────────

@bp.route("/staff_credentials", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def staff_credentials():
    """
    HR view: list and add professional licence / credential records.

    The StaffCredential model and admin API exist, but HR had no UI to view
    or enter credentials for staff they manage. This fills that gap.
    """

    try:
        if request.method == "POST":
            staff_name = (request.form.get("staff_name") or "").strip()
            employee_id_raw = request.form.get("employee_id") or None
            credential_type = (request.form.get("credential_type") or "").strip()
            credential_number = (request.form.get("credential_number") or "").strip()
            issue_date_raw = request.form.get("issue_date") or None
            expiry_date_raw = request.form.get("expiry_date") or None

            if not all([staff_name, credential_type, credential_number, expiry_date_raw]):
                raise ValueError("Staff name, credential type, number, and expiry date are required.")

            issue_date = datetime.strptime(issue_date_raw, "%Y-%m-%d").date() if issue_date_raw else None
            expiry_date = datetime.strptime(expiry_date_raw, "%Y-%m-%d").date()
            employee_id = int(employee_id_raw) if employee_id_raw else None

            cred = StaffCredential(
                employee_id=employee_id,
                staff_name=staff_name,
                credential_type=credential_type,
                credential_number=credential_number,
                issue_date=issue_date,
                expiry_date=expiry_date,
            )
            db.session.add(cred)
            _write_audit("Add Credential", {"staff": staff_name, "type": credential_type, "number": credential_number})
            db.session.commit()
            flash(f"Credential for {staff_name} added successfully.", "success")
            return redirect(url_for("hr.staff_credentials"))

        today = datetime.now(timezone.utc).date()
        credentials = StaffCredential.query.order_by(StaffCredential.expiry_date.asc()).all()
        employees = Employee.query.filter_by(is_active=True).order_by(Employee.name).all()

        return render_template(
            "hr/staff_credentials.html",
            credentials=credentials,
            employees=employees,
            today=today,
        )

    except (SQLAlchemyError, ValueError) as e:
        flash(str(e) if isinstance(e, ValueError) else "Something went wrong.", "error")
        logger.error("hr.staff_credentials failed: %s", e, exc_info=True)
        db.session.rollback()
        return redirect(url_for("hr.staff_credentials"))


# ── Performance Reviews ────────────────────────────────────────────────────

@bp.route("/performance_reviews")
@login_required
@roles_required("hr", "admin")
def performance_reviews():
    """List all performance reviews."""

    reviews = (
        PerformanceReview.query
        .join(Employee)
        .order_by(PerformanceReview.created_at.desc())
        .all()
    )
    return render_template("hr/performance_reviews.html", reviews=reviews)


@bp.route("/performance_review/new/<int:employee_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def new_performance_review(employee_id):
    """Create a new performance review for an employee."""

    employee = Employee.query.get_or_404(employee_id)
    form = PerformanceReviewForm()

    if form.validate_on_submit():
        try:
            review = PerformanceReview(
                employee_id=employee.id,
                reviewer_id=current_user.id,
                review_period=form.review_period.data,
                review_type=form.review_type.data,
                score=form.score.data,
                strengths=form.strengths.data,
                areas_for_improvement=form.areas_for_improvement.data,
                goals_next_period=form.goals_next_period.data,
                comments=form.comments.data,
            )
            db.session.add(review)
            _write_audit("Performance Review", {
                "employee_id": employee.employee_id,
                "period": form.review_period.data,
                "score": form.score.data,
            })
            db.session.commit()
            flash(f"Performance review for {employee.name} saved.", "success")
            return redirect(url_for("hr.employee_profile", employee_id=employee.id))
        except SQLAlchemyError as e:
            flash("Something went wrong.", "error")
            logger.error("hr.new_performance_review failed: %s", e, exc_info=True)
            db.session.rollback()

    return render_template("hr/new_performance_review.html", employee=employee, form=form)


# ── Training Records ───────────────────────────────────────────────────────

@bp.route("/training_records")
@login_required
@roles_required("hr", "admin")
def training_records():
    """List all training records."""

    records = (
        TrainingRecord.query
        .join(Employee)
        .order_by(TrainingRecord.date_completed.desc())
        .all()
    )
    return render_template("hr/training_records.html", records=records)


@bp.route("/training_record/new/<int:employee_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def new_training_record(employee_id):
    """Log a new training / CPD record for an employee."""

    employee = Employee.query.get_or_404(employee_id)
    form = TrainingRecordForm()

    if form.validate_on_submit():
        try:
            record = TrainingRecord(
                employee_id=employee.id,
                title=form.title.data,
                provider=form.provider.data,
                training_type=form.training_type.data,
                date_completed=form.date_completed.data,
                expiry_date=form.expiry_date.data,
                cpd_points=form.cpd_points.data,
                certificate_number=form.certificate_number.data,
                notes=form.notes.data,
                recorded_by=current_user.id,
            )
            db.session.add(record)
            _write_audit("Add Training Record", {
                "employee_id": employee.employee_id,
                "title": form.title.data,
            })
            db.session.commit()
            flash(f"Training record for {employee.name} saved.", "success")
            return redirect(url_for("hr.employee_profile", employee_id=employee.id))
        except SQLAlchemyError as e:
            flash("Something went wrong.", "error")
            logger.error("hr.new_training_record failed: %s", e, exc_info=True)
            db.session.rollback()

    return render_template("hr/new_training_record.html", employee=employee, form=form)


# ── Disciplinary Records ───────────────────────────────────────────────────

@bp.route("/disciplinary_records")
@login_required
@roles_required("hr", "admin")
def disciplinary_records():
    """List all disciplinary records."""

    records = (
        DisciplinaryRecord.query
        .filter_by(is_active=True)
        .join(Employee)
        .order_by(DisciplinaryRecord.incident_date.desc())
        .all()
    )
    return render_template("hr/disciplinary_records.html", records=records)


@bp.route("/disciplinary_record/new/<int:employee_id>", methods=["GET", "POST"])
@login_required
@roles_required("hr", "admin")
def new_disciplinary_record(employee_id):
    """Record a new disciplinary action against an employee."""

    employee = Employee.query.get_or_404(employee_id)
    form = DisciplinaryRecordForm()

    if form.validate_on_submit():
        try:
            record = DisciplinaryRecord(
                employee_id=employee.id,
                incident_date=form.incident_date.data,
                incident_type=form.incident_type.data,
                description=form.description.data,
                action_taken=form.action_taken.data,
                outcome=form.outcome.data or None,
                reviewed_by=current_user.id,
            )
            db.session.add(record)
            _write_audit("Disciplinary Record", {
                "employee_id": employee.employee_id,
                "type": form.incident_type.data,
            })
            db.session.commit()
            flash(f"Disciplinary record for {employee.name} saved.", "success")
            return redirect(url_for("hr.employee_profile", employee_id=employee.id))
        except SQLAlchemyError as e:
            flash("Something went wrong.", "error")
            logger.error("hr.new_disciplinary_record failed: %s", e, exc_info=True)
            db.session.rollback()

    return render_template("hr/new_disciplinary_record.html", employee=employee, form=form)
