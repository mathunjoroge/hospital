
from flask import flash, redirect, render_template, request, url_for
from flask_login import login_required
from flask_socketio import SocketIO
from sqlalchemy.orm import joinedload

from departments.models.laboratory import LabResult, LabResultTemplate
from departments.models.medicine import LabTest, RequestedLab
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

socketio = SocketIO()
 # Generate a UUID and convert it to a string

# Display the lab waiting list

@bp.route('/')
@login_required
@roles_required('laboratory', 'admin')
def index():
    """Displays the laboratory waiting list with pending lab test requests."""

    try:
        # Fetch all requested lab tests with status=0 (pending)
        pending_lab_requests = RequestedLab.query.filter_by(status=0).options(
            joinedload(RequestedLab.patient),  # Include patient details
            joinedload(RequestedLab.lab_test)  # Include lab test details
        ).all()

        # Debug: Print pending lab requests
        print("Pending Lab Requests:", pending_lab_requests)

        # If no pending requests exist, inform the user
        if not pending_lab_requests:
            flash('No pending lab test requests at the moment.', 'info')

        return render_template(
            'laboratory/index.html',
            pending_lab_requests=pending_lab_requests
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.index: {e}")  # Debugging
        return redirect(url_for('login'))
#display available lab tests
@bp.route('/lab_tests', methods=['GET'])
@login_required
@roles_required('laboratory', 'admin')
def lab_tests():
    """Displays a list of available lab tests."""

    try:
        # Fetch all lab tests from the database
        lab_tests = LabTest.query.all()

        # Render the lab_tests.html template with the fetched data
        return render_template(
            'laboratory/lab_tests.html',
            lab_tests=lab_tests
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.lab_tests: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))

#edit labtest
@bp.route('/edit_lab_test/<int:test_id>', methods=['GET', 'POST'])
@login_required
@roles_required('laboratory', 'admin', 'medicine')
def edit_lab_test(test_id):
    """Handles editing a specific lab test."""

    try:
        # Fetch the lab test by ID
        lab_test = LabTest.query.get_or_404(test_id)

        # Fetch all associated result templates
        result_templates = LabResultTemplate.query.filter_by(test_id=test_id).all()

        if request.method == 'POST':
            # Update lab test details
            lab_test.test_name = request.form.get('test_name', lab_test.test_name)
            lab_test.cost = float(request.form.get('cost', lab_test.cost))
            lab_test.description = request.form.get('description', lab_test.description)

            # Process updated parameters
            updated_parameters = {}
            for key, value in request.form.items():
                if key.startswith("parameter_name_"):
                    param_id = key.split("_")[-1]
                    parameter_name = value.strip()
                    normal_range_low = request.form.get(f"normal_range_low_{param_id}", "").strip()
                    normal_range_high = request.form.get(f"normal_range_high_{param_id}", "").strip()
                    unit = request.form.get(f"unit_{param_id}", "").strip()

                    # Store updated parameter data only if parameter_name is provided
                    if parameter_name:  # Ensure parameter_name is not empty
                        updated_parameters[param_id] = {
                            "parameter_name": parameter_name,
                            "normal_range_low": float(normal_range_low) if normal_range_low else None,
                            "normal_range_high": float(normal_range_high) if normal_range_high else None,
                            "unit": unit
                        }

            # Update existing result templates or delete if empty
            for template in result_templates:
                param_data = updated_parameters.get(str(template.id), {})
                if param_data.get("parameter_name"):  # If parameter name is not empty
                    template.parameter_name = param_data["parameter_name"]
                    template.normal_range_low = param_data["normal_range_low"]
                    template.normal_range_high = param_data["normal_range_high"]
                    template.unit = param_data["unit"]
                else:  # Delete the parameter if no name is provided
                    db.session.delete(template)

            # Add new parameters (if any)
            new_param_count = int(request.form.get('new_param_count', 0))
            for i in range(1, new_param_count + 1):
                new_parameter_name = request.form.get(f"new_parameter_name_{i}").strip()
                new_normal_range_low = request.form.get(f"new_normal_range_low_{i}", "").strip()
                new_normal_range_high = request.form.get(f"new_normal_range_high_{i}", "").strip()
                new_unit = request.form.get(f"new_unit_{i}", "").strip()

                if new_parameter_name:  # Add only if parameter_name is not empty
                    new_template = LabResultTemplate(
                        test_id=test_id,
                        parameter_name=new_parameter_name,
                        normal_range_low=float(new_normal_range_low) if new_normal_range_low else None,
                        normal_range_high=float(new_normal_range_high) if new_normal_range_high else None,
                        unit=new_unit
                    )
                    db.session.add(new_template)

            # Commit changes to the database
            db.session.commit()

            flash('Lab test updated successfully!', 'success')
            return redirect(url_for('laboratory.view_lab_test', test_id=test_id))

        # Render the edit form on GET request
        return render_template(
            'laboratory/edit_lab_test.html',
            lab_test=lab_test,
            result_templates=result_templates
        )

    except ValueError as ve:
        # Handle conversion errors (e.g., empty strings or invalid floats)
        flash(f'Invalid input: {ve}', 'error')
        print(f"Debug: Error in laboratory.edit_lab_test: {ve}")
        return redirect(url_for('laboratory.edit_lab_test', test_id=test_id))

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in laboratory.edit_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
#delete test
@bp.route('/delete_lab_test/<test_id>', methods=['POST'])
@login_required
@roles_required('laboratory', 'admin')
def delete_lab_test(test_id):
    """Handles deleting a lab test."""

    try:
        # Fetch the lab test by ID
        lab_test = LabTest.query.get_or_404(test_id)

        # Delete the lab test
        db.session.delete(lab_test)
        db.session.commit()

        flash('Lab test deleted successfully!', 'success')
        return redirect(url_for('laboratory.lab_tests'))

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.delete_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
#add lab test
@bp.route('/add_lab_test', methods=['GET', 'POST'])
@login_required
@roles_required('laboratory', 'admin')
def add_lab_test():
    """Handles adding a new lab test."""

    try:
        if request.method == 'POST':
            # Extract form data
            test_name = request.form.get('test_name')
            cost = request.form.get('cost')
            description = request.form.get('description')

            # Validation
            if not all([test_name, cost]):
                flash('Test name and cost are required!', 'error')
                return render_template('laboratory/add_lab_test.html')

            # Create a new lab test
            new_lab_test = LabTest(
                test_name=test_name,
                cost=float(cost),
                description=description
            )
            db.session.add(new_lab_test)
            db.session.commit()

            flash('Lab test added successfully!', 'success')
            return redirect(url_for('laboratory.lab_tests'))

        # Render the add form on GET request
        return render_template('laboratory/add_lab_test.html')

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.add_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))


@bp.route('/view_lab_test/<int:test_id>', methods=['GET'])
@login_required
@roles_required('laboratory', 'admin', 'medicine')
def view_lab_test(test_id):
    """Displays detailed information about a specific lab test."""

    try:
        # Perform the join and fetch the required columns
        lab_test_details = db.session.query(
            LabTest.test_name,
            LabTest.description,
            LabResultTemplate.parameter_name,
            LabResultTemplate.normal_range_low,
            LabResultTemplate.normal_range_high,
            LabResultTemplate.unit
        ).join(
            LabResultTemplate, LabTest.id == LabResultTemplate.test_id
        ).filter(
            LabTest.id == test_id
        ).all()

        if not lab_test_details:
            flash(f'Lab test with ID {test_id} not found!', 'error')
            return redirect(url_for('laboratory.lab_tests'))

        # Extract the test name and description (they are the same for all rows in the result)
        test_name = lab_test_details[0].test_name if lab_test_details else "Unknown Test"
        description = lab_test_details[0].description if lab_test_details else "No description provided"

        # Render the view_lab_test.html template with the fetched data
        return render_template(
            'laboratory/view_lab_test.html',
            test_name=test_name,
            description=description,
            parameters=lab_test_details  # Pass the list of parameters
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.view_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
#process lab results

@bp.route('/dashboard')
@login_required
@roles_required('laboratory', 'admin')
def dashboard():
    """Displays key lab statistics and trends."""

    total_tests = LabResult.query.count()
    pending_tests = RequestedLab.query.filter_by(status=0).count()
    abnormal_tests = LabResult.query.filter(LabResult.result.ilike('%abnormal%')).count()

    return render_template('laboratory/dashboard.html',
                           total_tests=total_tests,
                           pending_tests=pending_tests,
                           abnormal_tests=abnormal_tests)


