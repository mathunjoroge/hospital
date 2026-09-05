import json
import uuid
from datetime import datetime

from flask import Response, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required
from flask_socketio import SocketIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sqlalchemy.orm import joinedload

from departments.models.laboratory import LabResult, LabResultTemplate
from departments.models.medicine import LabTest, RequestedLab
from departments.models.records import Patient
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

socketio = SocketIO()
 # Generate a UUID and convert it to a string

# Display the lab waiting list

@bp.route('/process_lab_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
@roles_required('laboratory', 'admin')
def process_lab_request(request_id):
    """Handles processing a lab test request."""

    try:
        # Fetch the requested lab test by ID
        lab_request = RequestedLab.query.get_or_404(request_id)

        # Fetch the associated lab test details
        lab_test = LabTest.query.get_or_404(lab_request.lab_test_id)

        # Fetch all parameters (templates) for the lab test
        parameters = LabResultTemplate.query.filter_by(test_id=lab_test.id).all()

        if request.method == 'POST':
            # Extract form data for lab_test_id[] and result[]
            lab_test_ids = request.form.getlist('lab_test_id[]')
            results = request.form.getlist('result[]')

            # Combine lab_test_ids and results into a dictionary
            results_dict = {param: result.strip() for param, result in zip(lab_test_ids, results) if result.strip()}

            # Debug: Print raw form data and combined dictionary
            print(f"Debug: Raw Lab Test IDs: {lab_test_ids}")
            print(f"Debug: Raw Results: {results}")
            print(f"Debug: Combined Results Dictionary: {results_dict}")

            # Validate that all required fields are provided
            if not results_dict:
                flash('At least one result must be entered!', 'error')
                return render_template(
                    'laboratory/process_lab_request.html',
                    lab_request=lab_request,
                    lab_test=lab_test,
                    parameters=parameters,
                    result_id=session.get('result_id')  # Pass the result_id to the template
                )

            # Generate a unique result_id (if not already generated)
            result_id = session.get('result_id') or str(uuid.uuid4())
            session['result_id'] = result_id  # Store in session for consistency

            # Create or update the lab result record
            lab_result = LabResult(
                patient_id=lab_request.patient_id,
                lab_test_id=lab_test.id,
                test_date=datetime.utcnow(),
                result_notes=request.form.get('result_notes', ''),  # Optional notes
                result=json.dumps(results_dict),  # Store the results as a JSON string
                result_id=result_id,  # Assign the unique result_id
                updated_by=current_user.id  # Set the user who processed the result
            )

            db.session.add(lab_result)
            db.session.commit()

            # Update the lab request status to processed (e.g., status=1)
            lab_request.status = 1
            lab_request.result_id = result_id  # Link the lab request to the result via result_id
            db.session.commit()

            flash('Lab test results submitted successfully!', 'success')
            return redirect(url_for('laboratory.index'))  # Redirect back to the lab index page

        # Generate a unique result_id for the form (only on GET requests)
        if request.method == 'GET':
            session['result_id'] = str(uuid.uuid4())  # Store in session

        # Render the form on GET request
        return render_template(
            'laboratory/process_lab_request.html',
            lab_request=lab_request,
            lab_test=lab_test,
            parameters=parameters,
            result_id=session.get('result_id')  # Pass the result_id to the template
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in laboratory.process_lab_request: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))
#view lab results
@bp.route('/view_lab_results/<int:result_id>', methods=['GET'])
@login_required
@roles_required('laboratory', 'admin')
def view_lab_results(result_id):
    """Displays lab test results in a structured format."""

    try:
        # Fetch the lab result by ID
        lab_result = LabResult.query.get_or_404(result_id)

        # Debug: Print raw result string
        print(f"Debug: Raw Result String: {lab_result.result}")

        # Parse the result string back into a dictionary
        try:
            results_dict = json.loads(lab_result.result) if lab_result.result else {}
            print(f"Debug: Parsed Results Dictionary: {results_dict}")  # Debug
        except json.JSONDecodeError:
            flash('Something went wrong. Please try again.', 'warning')
            results_dict = {}  # Fallback to an empty dictionary if parsing fails

        # Fetch the associated lab test details
        lab_test = LabTest.query.get_or_404(lab_result.lab_test_id)

        # Fetch all parameters (templates) for the lab test
        parameters = LabResultTemplate.query.filter_by(test_id=lab_test.id).all()

        # Match results with parameter IDs and prepare presentation data
        test_presentation = []
        for param in parameters:
            result_value = results_dict.get(str(param.id))  # Ensure key is a string
            print(f"Debug: Parameter ID: {param.id}, Parameter Name: {param.parameter_name}, Result Value: {result_value}")  # Debugging

            try:
                # Convert result to float for comparisons
                result_value_float = float(result_value) if result_value is not None else None
            except ValueError:
                result_value_float = None  # Handle invalid data

            # Determine the status
            if result_value_float is None:
                status = "Invalid Result"
            elif result_value_float < param.normal_range_low:
                status = "Low"
            elif result_value_float > param.normal_range_high:
                status = "High"
            else:
                status = "Normal"

            # Append to test_presentation
            test_presentation.append({
                'parameter_name': param.parameter_name,
                'normal_range_low': param.normal_range_low,
                'normal_range_high': param.normal_range_high,
                'unit': param.unit,
                'result': result_value if result_value is not None else "N/A",  # Handle missing results
                'status': status  # Set final status
            })

        # Debug: Print final test presentation data
        print(f"Debug: Final Test Presentation Data: {test_presentation}")

        return render_template(
            'laboratory/view_lab_results.html',
            lab_test=lab_test,
            lab_result=lab_result,
            test_presentation=test_presentation
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.view_lab_results: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))

@bp.route('/pending_lab_results')
@login_required
@roles_required('laboratory', 'admin')
def pending_lab_results():
    """Displays pending lab test requests."""

    try:
        # Fetch all pending lab test requests
        pending_lab_requests = RequestedLab.query.filter_by(status=0).options(
            joinedload(RequestedLab.patient),
            joinedload(RequestedLab.lab_test)
        ).all()

        return render_template(
            'laboratory/pending_lab_results.html',
            pending_lab_requests=pending_lab_requests
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.pending_lab_results: {e}")
        return redirect(url_for('laboratory.index'))
@bp.route('/processed_lab_results')
@login_required
@roles_required('laboratory', 'admin')
def processed_lab_results():
    """Displays processed lab test results."""

    try:
        # Fetch processed lab results where updated_by is NOT NULL
        processed_lab_results = db.session.query(
            LabResult.id,
            LabResult.patient_id,
            LabResult.lab_test_id,
            LabResult.test_date,
            LabResult.result_id,
            LabResult.result_notes,
            LabResult.updated_by,
            LabTest.test_name,
            Patient.name.label("patient_name")
        ).join(LabTest, LabTest.id == LabResult.lab_test_id) \
         .join(Patient, Patient.patient_id == LabResult.patient_id) \
         .filter(LabResult.updated_by.isnot(None)) \
         .order_by(LabResult.test_date.desc()) \
         .all()

        # Debug: Print processed results
        print("Processed Lab Results:", processed_lab_results)

        # If no processed results exist, inform the user
        if not processed_lab_results:
            flash('No processed lab results available.', 'info')

        return render_template(
            'laboratory/processed_lab_results.html',
            processed_lab_results=processed_lab_results
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.processed_lab_results: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))


@bp.route('/generate_lab_report/<int:result_id>')
@login_required
@roles_required('laboratory', 'admin')
def generate_lab_report(result_id):
    """Generates a PDF lab test report."""

    result = LabResult.query.get_or_404(result_id)

    response = Response(content_type='application/pdf')
    response.headers["Content-Disposition"] = f"inline; filename=Lab_Report_{result_id}.pdf"

    pdf = canvas.Canvas(response.stream, pagesize=letter)
    pdf.drawString(100, 750, f"Lab Report for {result.patient.name} - {result.test_date.strftime('%Y-%m-%d')}")
    pdf.drawString(100, 730, f"Test: {result.lab_test.test_name}")
    pdf.drawString(100, 710, f"Results: {result.result}")

    pdf.showPage()
    pdf.save()

    return response
@bp.route('/search-patient', methods=['GET', 'POST'])
@login_required
@roles_required('laboratory', 'admin')
def search_patient():
    """Search for a patient and fetch their lab history on the same page."""

    patients = []
    lab_results = []
    selected_patient = None  # Track which patient is selected

    if request.method == 'POST':
        search_query = request.form.get('search_query')

        if search_query:
            # Search for patients by ID or Name (case insensitive)
            patients = Patient.query.filter(
                (Patient.patient_id.ilike(f"%{search_query}%")) |
                (Patient.name.ilike(f"%{search_query}%"))
            ).all()

            if patients:
                # Only fetch lab history if a patient is selected
                selected_patient_id = request.form.get('selected_patient_id')

                if selected_patient_id:
                    selected_patient = Patient.query.filter_by(patient_id=selected_patient_id).first()

                    if selected_patient:
                        lab_results = db.session.query(
                            LabResult.id,
                            LabResult.test_date,
                            LabResult.result_id,
                            LabResult.result_notes,
                            LabTest.test_name
                        ).join(LabTest, LabResult.lab_test_id == LabTest.id) \
                         .filter(LabResult.patient_id == selected_patient.patient_id) \
                         .order_by(LabResult.test_date.desc()).all()
            else:
                flash("No patients found with that name or ID.", "warning")

    return render_template(
        'laboratory/search_patient.html',
        patients=patients,
        lab_results=lab_results,
        selected_patient=selected_patient
    )


#abnormal results
@bp.route('/abnormal_results')
@login_required
@roles_required('laboratory', 'admin')
def abnormal_results():
    """Displays lab test results that are outside normal ranges."""

    try:
        abnormal_results = db.session.query(
            LabResult.id,
            LabResult.patient_id,
            LabResult.lab_test_id,
            LabResult.result,
            LabTest.test_name
        ).join(LabTest, LabResult.lab_test_id == LabTest.id).all()

        flagged_results = []
        for result in abnormal_results:
            try:
                result_data = json.loads(result.result)  # Convert stored JSON result back to dictionary
                abnormal_parameters = []

                # Check each parameter against its normal range
                for param_id, value in result_data.items():
                    param = LabResultTemplate.query.get(param_id)
                    if param and (float(value) < param.normal_range_low or float(value) > param.normal_range_high):
                        abnormal_parameters.append({
                            'parameter': param.parameter_name,
                            'value': value,
                            'normal_range': f"{param.normal_range_low} - {param.normal_range_high}",
                            'unit': param.unit
                        })

                if abnormal_parameters:
                    flagged_results.append({
                        'id': result.id,
                        'patient_id': result.patient_id,
                        'test_name': result.test_name,
                        'abnormal_parameters': abnormal_parameters
                    })

            except Exception as e:
                print(f"Error processing lab result {result.id}: {e}")

        return render_template('laboratory/abnormal_results.html', flagged_results=flagged_results)

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in laboratory.abnormal_results: {e}")
        return redirect(url_for('laboratory.index'))
#dashboard
