from flask import render_template, redirect, url_for, request, flash, make_response
from flask_login import login_required, current_user
from . import bp  # Import the blueprint
from departments.models.hr import Employee, Rota, Payroll,Allowance,Deduction,Leave,CustomRule,AuditLog# Import Employee model
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from extensions import db
from datetime import datetime, timedelta
from collections import defaultdict
import random
from flask_mail import Message, Mail
import csv
from io import StringIO
from flask import send_file
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from departments.forms import AddAllowanceForm, AddDeductionForm, LeaveRequestForm,UpdateProfileForm # Import AddAllowanceForm and AddDeductionForm

@bp.route('/', methods=['GET'])
@login_required
def index():
    """HR dashboard displaying key metrics and recent changes."""
    if current_user.role not in ['hr', 'admin']:
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

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
            {"description": "Employee E0001 marked as inactive.", "date": datetime.now() - timedelta(days=1)},
            {"description": "New employee E0002 added to the system.", "date": datetime.now() - timedelta(days=2)}
        ]  # Replace with actual query logic

        return render_template(
            'hr/index.html',
            active_employees_count=active_employees_count,
            inactive_employees_count=inactive_employees_count,
            total_employees_count=total_employees_count,
            recent_changes=recent_changes
        )

    except Exception as e:
        flash(f'Error loading HR dashboard: {e}', 'error')
        print(f"Debug: Error in hr.index: {e}")
        return redirect(url_for('home'))

@bp.route('/employee_list', methods=['GET'])
@login_required
def employee_list():
    """Displays the list of all employees."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all employees
        employees = Employee.query.order_by(Employee.date_hired.desc()).all()
        return render_template('hr/employee_list.html', employees=employees)

    except Exception as e:
        flash(f'Error fetching employee list: {e}', 'error')
        print(f"Debug: Error in hr.employee_list: {e}")
        return redirect(url_for('home'))

@bp.route('/new_employee', methods=['GET', 'POST'])
@login_required
def new_employee():
    """Registers a new employee with allowances and deductions."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        if request.method == 'POST':
            # Extract form data
            name = request.form.get('name')
            role = request.form.get('role')
            department = request.form.get('department')
            job_group = request.form.get('job_group')
            allowances = request.form.getlist('allowances')  # Selected allowances
            deductions = request.form.getlist('deductions')  # Selected deductions

            # Validate input
            if not all([name, role, department, job_group]):
                raise ValueError("All fields are required!")

            # Generate a unique employee ID
            employee_id = Employee.generate_employee_id()

            # Create a new employee entry
            new_employee = Employee(
                employee_id=employee_id,
                name=name,
                role=role,
                department=department,
                job_group=job_group
            )
            db.session.add(new_employee)

            # Assign allowances
            selected_allowances = Allowance.query.filter(Allowance.id.in_(allowances)).all()
            new_employee.allowances.extend(selected_allowances)

            # Assign deductions
            for deduction_id in deductions:
                deduction = Deduction.query.get(deduction_id)
                if deduction:
                    new_deduction = Deduction(
                        employee_id=new_employee.employee_id,
                        name=deduction.name,
                        type=deduction.type,
                        value=deduction.value
                    )
                    db.session.add(new_deduction)

            db.session.commit()
            flash(f'Employee {name} added successfully!', 'success')
            return redirect(url_for('hr.employee_list'))

        # Fetch all allowances and deductions
        allowances = Allowance.query.all()
        deductions = Deduction.query.all()

        return render_template('hr/new_employee.html', allowances=allowances, deductions=deductions)

    except Exception as e:
        flash(f'Error adding employee: {e}', 'error')
        print(f"Debug: Error in hr.new_employee: {e}")
        db.session.rollback()
        return redirect(url_for('hr.index'))

@bp.route('/update_employee/<int:employee_id>', methods=['GET', 'POST'])
@login_required
def update_employee(employee_id):
    """Updates an existing employee."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the employee by ID
        employee = Employee.query.get_or_404(employee_id)

        if request.method == 'POST':
            # Extract form data
            name = request.form.get('name')
            role = request.form.get('role')
            department = request.form.get('department')
            is_active = request.form.get('is_active') == 'on'  # Checkbox handling

            # Validate input
            if not all([name, role, department]):
                raise ValueError("All fields are required!")

            # Update employee details
            employee.name = name
            employee.role = role
            employee.department = department
            employee.is_active = is_active
            employee.updated_by = current_user.id

            db.session.commit()
            flash(f'Employee {name} updated successfully!', 'success')
            return redirect(url_for('hr.employee_list'))

        return render_template('hr/update_employee.html', employee=employee)

    except Exception as e:
        flash(f'Error updating employee: {e}', 'error')
        print(f"Debug: Error in hr.update_employee: {e}")
        db.session.rollback()
        return redirect(url_for('hr.employee_list'))

@bp.route('/delete_employee/<int:employee_id>', methods=['POST'])
@login_required
def delete_employee(employee_id):
    """Deletes an employee from the system."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the employee by ID
        employee = Employee.query.get_or_404(employee_id)

        # Delete the employee
        db.session.delete(employee)
        db.session.commit()

        flash(f'Employee {employee.name} deleted successfully!', 'success')
        return redirect(url_for('hr.employee_list'))

    except Exception as e:
        flash(f'Error deleting employee: {e}', 'error')
        print(f"Debug: Error in hr.delete_employee: {e}")
        db.session.rollback()
        return redirect(url_for('hr.employee_list'))
@bp.route('/rota_management', methods=['GET', 'POST'])
@login_required
def rota_management():
    """Manage employee shifts and schedules (rota)."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all active employees
        employees = Employee.query.filter_by(is_active=True).all()

        # Fetch all rotas
        rotas = Rota.query.order_by(Rota.week_range.desc()).all()

        # Prepare rota data for rendering
        rota_data = defaultdict(lambda: defaultdict(list))
        for rota in rotas:
            if rota.shift_8_5:
                rota_data[rota.week_range]['morning'] = [name.strip() for name in rota.shift_8_5.split(',')]
            if rota.shift_5_8:
                rota_data[rota.week_range]['evening'] = [rota.shift_5_8.strip()]
            if rota.shift_8_8:
                rota_data[rota.week_range]['night'] = [rota.shift_8_8.strip()]

        if request.method == 'POST':
            # Extract form data
            week_range = request.form.get('week_range')  # Selected week range
            if not week_range:
                raise ValueError("Week range is required!")

            # Automatically allocate shifts
            start_date, end_date = week_range.split(' - ')
            start_date = datetime.strptime(start_date.strip(), '%d/%m/%Y').date()
            end_date = datetime.strptime(end_date.strip(), '%d/%m/%Y').date()

            # Ensure the week range is valid
            if (end_date - start_date).days != 6:
                raise ValueError("Invalid week range! Please specify a 7-day period.")

            # Allocate shifts
            morning_shifts = []
            evening_shifts = []
            night_shifts = []

            # Randomly assign shifts while respecting constraints
            for emp in employees:
                shift_options = ['morning', 'evening', 'night']
                assigned_shift = random.choice(shift_options)

                # Prevent consecutive night/evening shifts
                recent_shifts = Rota.query.filter(
                    Rota.week_range >= (start_date - timedelta(days=7)),
                    Rota.week_range <= (start_date + timedelta(days=7))
                ).all()

                recent_morning = any(emp.name in r.shift_8_5.split(',') if r.shift_8_5 else False for r in recent_shifts)
                recent_evening = any(emp.name == r.shift_5_8 if r.shift_5_8 else False for r in recent_shifts)
                recent_night = any(emp.name == r.shift_8_8 if r.shift_8_8 else False for r in recent_shifts)

                # Apply constraints
                if assigned_shift == 'morning' and not recent_morning:
                    morning_shifts.append(emp.name)
                elif assigned_shift == 'evening' and not recent_evening:
                    evening_shifts.append(emp.name)
                elif assigned_shift == 'night' and not recent_night:
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
            flash(f'Rota for {week_range} generated successfully!', 'success')
            return redirect(url_for('hr.rota_management'))

        return render_template(
            'hr/rota_management.html',
            employees=employees,
            rotas=rotas,
            rota_data=dict(rota_data)  # Convert defaultdict to dict for Jinja2
        )

    except Exception as e:
        flash(f'Error generating rota: {e}', 'error')
        print(f"Debug: Error in hr.rota_management: {e}")
        db.session.rollback()
        return redirect(url_for('hr.index'))


@bp.route('/department_reports', methods=['GET'])
@login_required
def department_reports():
    """Generate department-wise employee distribution reports."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all active employees grouped by department and role
        filters = {}
        role_filter = request.args.get('role')  # Optional role filter from query parameters
        if role_filter:
            filters['role'] = role_filter

        employees_by_department = Employee.query.filter_by(is_active=True, **filters).all()

        # Group employees by department
        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]['total'] += 1
            department_data[emp.department][emp.role] += 1

        # Convert defaultdict to dict for Jinja2 rendering
        department_data = dict(department_data)

        # Debugging output
        print(f"Debug: Department-wise employee distribution: {department_data}")

        return render_template(
            'hr/department_reports.html',
            department_data=department_data,
            role_filter=role_filter
        )

    except Exception as e:
        flash(f'Error generating department reports: {e}', 'error')
        print(f"Debug: Error in hr.department_reports: {e}")
        return redirect(url_for('hr.index'))

@bp.route('/export_department_reports', methods=['GET'])
@login_required
def export_department_reports():
    """Export department-wise employee distribution reports to CSV."""
    if current_user.role != 'hr':
        flash('Unauthorized access. HR staff only.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all active employees grouped by department and role
        employees_by_department = Employee.query.filter_by(is_active=True).all()

        # Prepare data for CSV
        department_data = defaultdict(lambda: defaultdict(int))
        for emp in employees_by_department:
            department_data[emp.department]['total'] += 1
            department_data[emp.department][emp.role] += 1

        # Convert data to list of rows
        csv_data = [["Department", "Total", "Records", "Nursing", "Pharmacy", "Medicine", "Laboratory", "Imaging", "Mortuary", "HR", "Stores", "Admin"]]
        for dept, counts in department_data.items():
            row = [
                dept,
                counts['total'],
                counts.get('records', 0),
                counts.get('nursing', 0),
                counts.get('pharmacy', 0),
                counts.get('medicine', 0),
                counts.get('laboratory', 0),
                counts.get('imaging', 0),
                counts.get('mortuary', 0),
                counts.get('hr', 0),
                counts.get('stores', 0),
                counts.get('admin', 0)
            ]
            csv_data.append(row)

        # Generate CSV response
        si = StringIO()
        writer = csv.writer(si)
        writer.writerows(csv_data)
        output = make_response(si.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=department_reports.csv"
        output.headers["Content-type"] = "text/csv"
        return output

    except Exception as e:
        flash(f'Error exporting department reports: {e}', 'error')
        print(f"Debug: Error in hr.export_department_reports: {e}")
        return redirect(url_for('hr.department_reports'))
@bp.route('/payroll')
def payroll_dashboard():
    search_query = request.args.get('search', '')
    month_filter = request.args.get('month', '')
    department_filter = request.args.get('department', '')

    query = Payroll.query.join(Employee)
    if search_query:
        query = query.filter(Employee.name.contains(search_query))
    if month_filter:
        query = query.filter(Payroll.month == month_filter)
    if department_filter:
        query = query.filter(Employee.department == department_filter)

    payrolls = query.all()
    return render_template('hr/dashboard.html', payrolls=payrolls)

@bp.route('/generate_payroll/<month>')
def generate_payroll(month):
    employees = Employee.query.filter_by(is_active=True).all()
    for employee in employees:
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
            net_pay=net_pay
        )
        db.session.add(payroll)
    db.session.commit()
    return "Payroll generated successfully!"
@bp.route('/employee_payroll/<int:employee_id>')
def employee_payroll(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    payrolls = Payroll.query.filter_by(employee_id=employee.id).all()
    return render_template('hr/employee_payroll.html', employee=employee, payrolls=payrolls)
@bp.route('/add_deduction', methods=['GET', 'POST'])
def add_deduction():
    form = AddDeductionForm()
    if form.validate_on_submit():
        deduction = Deduction(
            name=form.name.data,
            value=form.value.data,
            is_percentage=form.is_percentage.data
        )
        db.session.add(deduction)
        db.session.commit()
        flash('Deduction added successfully!', 'success')
        return redirect(url_for('payroll_dashboard'))
    return render_template('hr/add_deduction.html', form=form)

@bp.route('/add_allowance', methods=['GET', 'POST'])
def add_allowance():
    form = AddAllowanceForm()
    if form.validate_on_submit():
        allowance = Allowance(
            job_group=form.job_group.data,
            name=form.name.data,
            value=form.value.data
        )
        db.session.add(allowance)
        db.session.commit()
        flash('Allowance added successfully!', 'success')
        return redirect(url_for('payroll_dashboard'))
    return render_template('hr/add_allowance.html', form=form)

@bp.route('/leave_request', methods=['GET', 'POST'])
def leave_request():
    form = LeaveRequestForm()
    if form.validate_on_submit():
        leave = Leave(
            employee_id=1,  # Replace with actual employee ID
            start_date=form.start_date.data,
            end_date=form.end_date.data,
            type=form.type.data
        )
        db.session.add(leave)
        db.session.commit()
        flash('Leave request submitted successfully!', 'success')
        return redirect(url_for('index'))
    return render_template('hr/leave_request.html', form=form)

@bp.route('/hr/reports')
def reports():
    # Get filter parameters from the request
    month = request.args.get('month')
    employee_id = request.args.get('employee_id')
    department = request.args.get('department')

    # Build the query
    query = Payroll.query.join(Employee)
    if month:
        query = query.filter(Payroll.month == month)
    if employee_id:
        query = query.filter(Payroll.employee_id == employee_id)
    if department:
        query = query.filter(Employee.department == department)

    payrolls = query.all()
    return render_template('hr/reports.html', payrolls=payrolls)

@bp.route('/audit_logs')
def audit_logs():
    logs = AuditLog.query.all()
    return render_template('hr/audit_logs.html', logs=logs)

@bp.route('/employee_profile/<int:employee_id>')
def employee_profile(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    return render_template('employee_profile.html', employee=employee)

@bp.route('/update_profile/<int:employee_id>', methods=['POST'])
def update_profile(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    employee.name = request.form.get('name')
    employee.department = request.form.get('department')
    employee.job_group = request.form.get('job_group')
    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('hr.employee_profile', employee_id=employee.id))

@bp.route('/leave_management')
def leave_management():
    # Fetch all leave requests with employee details
    leaves = db.session.query(Leave, Employee).join(Employee).all()
    return render_template('hr/leave_management.html', leaves=leaves)



@bp.route('/reject_leave/<int:leave_id>')
def reject_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
    leave.status = 'Rejected'
    db.session.commit()
    flash('Leave request rejected successfully!', 'danger')
    return redirect(url_for('hr.leave_management'))

@bp.route('/update_profile', methods=['GET', 'POST'])
@login_required
def update_employee_profile():
    employee = Employee.query.get_or_404(current_user.id)
    form = UpdateProfileForm()

    if form.validate_on_submit():
        employee.email = form.email.data
        employee.phone = form.phone.data
        employee.bank_name = form.bank_name.data
        employee.bank_account = form.bank_account.data
        db.session.commit()
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('employee_profile'))

    # Pre-fill the form with existing data
    form.email.data = employee.email
    form.phone.data = employee.phone
    form.bank_name.data = employee.bank_name
    form.bank_account.data = employee.bank_account

    return render_template('hr/employee_profile.html', form=form)

@bp.route('/payslips')
@login_required
def employee_payslips():
    payrolls = Payroll.query.filter_by(employee_id=current_user.id).all()
    return render_template('hr/employee_payslips.html', payrolls=payrolls)

@bp.route('/export_payroll_pdf')
def export_payroll_pdf():
    payrolls = Payroll.query.join(Employee).all()

    # Create a PDF
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "Payroll Report")
    y = 730
    for payroll in payrolls:
        p.drawString(100, y, f"{payroll.employee.name} - {payroll.month}: ${payroll.net_pay}")
        y -= 20
    p.showPage()
    p.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='payroll_report.pdf', mimetype='application/pdf')

@bp.route('/export_payroll_excel')
def export_payroll_excel():
    payrolls = Payroll.query.join(Employee).all()

    # Create a DataFrame
    data = {
        'Employee': [payroll.employee.name for payroll in payrolls],
        'Month': [payroll.month for payroll in payrolls],
        'Gross Pay': [payroll.gross_pay for payroll in payrolls],
        'Deductions': [payroll.total_deductions for payroll in payrolls],
        'Net Pay': [payroll.net_pay for payroll in payrolls]
    }
    df = pd.DataFrame(data)

    # Save to Excel
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='payroll_report.xlsx', mimetype='application/vnd.ms-excel')
def send_email(subject, recipient, body):
    msg = Message(subject, recipients=[recipient])
    msg.body = body
    Mail.send(msg)

@bp.route('/approve_leave/<int:leave_id>')
def approve_leave(leave_id):
    leave = Leave.query.get_or_404(leave_id)
    leave.status = 'Approved'
    db.session.commit()

    # Access the employee associated with the leave request
    employee = Employee.query.get(leave.employee_id)  # Assuming `employee_id` is the foreign key in the `Leave` model

    # Send notification
    if employee:
        send_email(
            subject='Leave Request Approved',
            recipient=employee.email,  # Access the email through the `employee` object
            body=f'Your leave request from {leave.start_date} to {leave.end_date} has been approved.'
        )
    else:
        flash('Employee not found for this leave request!', 'warning')

    flash('Leave request approved successfully!', 'success')
    return redirect(url_for('hr.leave_management'))

@bp.route('/process_payroll')
def process_payroll():
    # Process payroll logic here...

    # Send notification
    employees = Employee.query.all()
    for employee in employees:
        send_email(
            subject='Payroll Processed',
            recipient=employee.email,
            body=f'Your payroll for the month has been processed. Net Pay: ${employee.payrolls[-1].net_pay}'
        )

    flash('Payroll processed successfully!', 'success')
    return redirect(url_for('hr.payroll_dashboard'))
@bp.route('/view_payslip/<int:payroll_id>')
def view_payslip(payroll_id):
    payroll = Payroll.query.get_or_404(payroll_id)
    return render_template('hr/payslip.html', payroll=payroll)
@bp.route('/download_payslip/<int:payroll_id>')
def download_payslip(payroll_id):
    payroll = Payroll.query.get_or_404(payroll_id)

    # Create a PDF payslip
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Company Logo
    logo_path = "static/images/logo.png"  # Adjust the path as necessary
    logo = Image(logo_path, width=1.5*inch, height=1*inch)
    elements.append(logo)
    elements.append(Spacer(1, 12))

    # Payslip Header
    elements.append(Paragraph("Payslip", styles['Title']))
    elements.append(Paragraph(f"{payroll.month}", styles['Heading2']))
    elements.append(Spacer(1, 12))

    # Employee Details
    employee_details = [
        ["Employee Name:", payroll.employee.name],
        ["Employee ID:", payroll.employee.employee_id],
        ["Department:", payroll.employee.department],
        ["Job Group:", payroll.employee.job_group],
        ["Date Hired:", payroll.employee.date_hired.strftime('%Y-%m-%d')],
        ["Status:", "Active" if payroll.employee.is_active else "Inactive"]
    ]
    employee_table = Table(employee_details, colWidths=[2*inch, 4*inch])
    employee_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(employee_table)
    elements.append(Spacer(1, 12))

    # Payroll Details
    payroll_details = [
        ["Gross Pay:", f"${payroll.gross_pay}"],
        ["Total Deductions:", f"${payroll.total_deductions}"],
        ["Net Pay:", f"${payroll.net_pay}"],
        ["Payment Date:", payroll.month]
    ]
    payroll_table = Table(payroll_details, colWidths=[2*inch, 4*inch])
    payroll_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(payroll_table)
    elements.append(Spacer(1, 12))

    # Footer
    elements.append(Paragraph("This is an official payslip generated by Your Company Name. For any discrepancies, please contact HR.", styles['BodyText']))
    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)

    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"payslip_{payroll.employee.name}_{payroll.month}.pdf", mimetype='application/pdf')