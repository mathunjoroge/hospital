from flask import  render_template, redirect, url_for, request, flash
from flask import Response
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask_login import login_required, current_user
from extensions import db
from flask import session
from datetime import datetime,date
import uuid
import json
from uuid import uuid4
from sqlalchemy.orm import joinedload
from . import bp  # Import the blueprint
import os 
from flask_socketio import SocketIO
from departments.models.records import PatientWaitingList,Patient
from departments.models.medicine import (LabTest,RequestedLab
)
from departments.models.stores import (NonPharmCategory,NonPharmItem,OtherOrder
)
from departments.models.laboratory import LabResultTemplate,LabResult
from departments.models.user import User
socketio = SocketIO()
 # Generate a UUID and convert it to a string

# Display the lab waiting list
@bp.route('/')
@login_required
def index():
    """Displays the laboratory waiting list with pending lab test requests."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

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
        flash(f'Error fetching pending lab test requests: {e}', 'error')
        print(f"Debug: Error in laboratory.index: {e}")  # Debugging
        return redirect(url_for('login'))
#display available lab tests
@bp.route('/lab_tests', methods=['GET'])
@login_required
def lab_tests():
    """Displays a list of available lab tests."""
    if current_user.role not in ['laboratory', 'admin']: 
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))  

    try:
        # Fetch all lab tests from the database
        lab_tests = LabTest.query.all()

        # Render the lab_tests.html template with the fetched data
        return render_template(
            'laboratory/lab_tests.html',
            lab_tests=lab_tests
        )

    except Exception as e:
        flash(f'Error fetching lab tests: {e}', 'error')
        print(f"Debug: Error in laboratory.lab_tests: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))
    
#edit labtest
@bp.route('/edit_lab_test/<int:test_id>', methods=['GET', 'POST'])
@login_required
def edit_lab_test(test_id):
    """Handles editing a specific lab test."""
    if current_user.role != 'medicine' and current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

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
        flash(f'Error editing lab test: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in laboratory.edit_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
#delete test
@bp.route('/delete_lab_test/<test_id>', methods=['POST'])
@login_required
def delete_lab_test(test_id):
    """Handles deleting a lab test."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

    try:
        # Fetch the lab test by ID
        lab_test = LabTest.query.get_or_404(test_id)

        # Delete the lab test
        db.session.delete(lab_test)
        db.session.commit()

        flash('Lab test deleted successfully!', 'success')
        return redirect(url_for('laboratory.lab_tests'))

    except Exception as e:
        flash(f'Error deleting lab test: {e}', 'error')
        print(f"Debug: Error in laboratory.delete_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests')) 
#add lab test
@bp.route('/add_lab_test', methods=['GET', 'POST'])
@login_required
def add_lab_test():
    """Handles adding a new lab test."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

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
        flash(f'Error adding lab test: {e}', 'error')
        print(f"Debug: Error in laboratory.add_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
    
from sqlalchemy.orm import joinedload

@bp.route('/view_lab_test/<int:test_id>', methods=['GET'])
@login_required
def view_lab_test(test_id):
    """Displays detailed information about a specific lab test."""
    if current_user.role != 'medicine' and current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

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
        flash(f'Error fetching lab test details: {e}', 'error')
        print(f"Debug: Error in laboratory.view_lab_test: {e}")
        return redirect(url_for('laboratory.lab_tests'))
#process lab results
@bp.route('/process_lab_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
def process_lab_request(request_id):
    """Handles processing a lab test request."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

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
        flash(f'Error processing lab test request: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in laboratory.process_lab_request: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))
#view lab results
@bp.route('/view_lab_results/<int:result_id>', methods=['GET'])
@login_required
def view_lab_results(result_id):
    """Displays lab test results in a structured format."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

    try:
        # Fetch the lab result by ID
        lab_result = LabResult.query.get_or_404(result_id)

        # Debug: Print raw result string
        print(f"Debug: Raw Result String: {lab_result.result}")

        # Parse the result string back into a dictionary
        try:
            results_dict = json.loads(lab_result.result) if lab_result.result else {}
            print(f"Debug: Parsed Results Dictionary: {results_dict}")  # Debug
        except json.JSONDecodeError as e:
            flash(f'Invalid result format: {e}. Results may not be displayed correctly.', 'warning')
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
        flash(f'Error fetching lab test results: {e}', 'error')
        print(f"Debug: Error in laboratory.view_lab_results: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))
    #order for reagets
@bp.route('/reagents_order', methods=['GET', 'POST'])
@login_required
def reagents_order():
    """Handles ordering of lab reagents."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

    try:
        if request.method == 'POST':
            # Extract form data
            item_id = request.form.get('item_id', type=int)
            quantity = request.form.get('quantity', type=int)
            notes = request.form.get('notes', '')

            # Validation: Ensure item selection and valid quantity
            if not item_id or not quantity or quantity <= 0:
                flash('Please select a valid reagent and enter a quantity greater than zero.', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Fetch the reagent from the inventory (Category ID = 6 for Lab Reagents)
            reagent = NonPharmItem.query.filter_by(id=item_id, category_id=6).first()

            if not reagent:
                flash('Selected item is not a valid lab reagent.', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Check stock availability
            if reagent.stock_level < quantity:
                flash(f'Insufficient stock for {reagent.name}. Available: {reagent.stock_level}', 'error')
                return redirect(url_for('laboratory.reagents_order'))

            # Create a new reagent order
            new_order = OtherOrder(
                request_date=date.today(),
                status='Pending',
                requested_by=current_user.id,
                item_id=item_id,
                quantity_requested=quantity,
                quantity_issued=0,
                notes=notes
            )
            db.session.add(new_order)
            db.session.commit()

            flash(f'Order for {reagent.name} added successfully!', 'success')
            return redirect(url_for('laboratory.reagents_order'))

        # Fetch all pending reagent orders
        pending_orders = OtherOrder.query.filter_by(status='Pending')\
                                        .options(joinedload(OtherOrder.item))\
                                        .order_by(OtherOrder.request_date.desc())\
                                        .all()

        # Fetch only reagents (category_id=6)
        lab_reagents = NonPharmItem.query.filter_by(category_id=6).order_by(NonPharmItem.name).all()

        # Fetch user details for pending orders
        user_ids = [order.requested_by for order in pending_orders]
        users = User.query.filter(User.id.in_(user_ids)).all()

        # Use `username` instead of `name`
        user_name_map = {user.id: user.username for user in users}

        # Debugging Logs
        print(f"Debug: Found {len(pending_orders)} pending orders")
        for order in pending_orders:
            requester_name = user_name_map.get(order.requested_by, 'Unknown')
            item_name = order.item.name if order.item else 'Unknown'
            print(f"Debug: Order ID {order.id}, Requested by {requester_name}, Item: {item_name}, Quantity: {order.quantity_requested}")

        return render_template(
            'laboratory/reagents_order.html',
            pending_orders=pending_orders,
            user_name_map=user_name_map,
            lab_reagents=lab_reagents
        )

    except Exception as e:
        db.session.rollback()
        flash(f'Error processing order: {e}', 'error')
        print(f"Debug: Error in laboratory.reagents_order: {e}")
        return redirect(url_for('laboratory.index'))

@bp.route('/pending_lab_results')
@login_required
def pending_lab_results():
    """Displays pending lab test requests."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

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
        flash(f'Error fetching pending lab results: {e}', 'error')
        print(f"Debug: Error in laboratory.pending_lab_results: {e}")
        return redirect(url_for('laboratory.index'))
@bp.route('/processed_lab_results')
@login_required
def processed_lab_results():
    """Displays processed lab test results."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('login'))

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
        flash(f'Error fetching processed lab results: {e}', 'error')
        print(f"Debug: Error in laboratory.processed_lab_results: {e}")  # Debugging
        return redirect(url_for('laboratory.index'))


#inventory, to be edited later
@bp.route('/lab_reagent_inventory')
@login_required
def lab_reagent_inventory():
    """Displays available lab reagents and stock levels."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all reagents where category_id = 6
        reagents = NonPharmItem.query.filter_by(category_id=6).order_by(NonPharmItem.name).all()

        return render_template(
            'laboratory/lab_reagent_inventory.html',
            reagents=reagents
        )

    except Exception as e:
        flash(f'Error fetching reagent inventory: {e}', 'error')
        print(f"Debug: Error in laboratory.lab_reagent_inventory: {e}")
        return redirect(url_for('laboratory.index'))
@bp.route('/request_reagent_restock', methods=['POST'])
@login_required
def request_reagent_restock():
    """Handles reagent restock requests."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        item_id = request.form.get('item_id')
        quantity = request.form.get('quantity')

        if not item_id or not quantity:
            flash("Item and quantity required!", "error")
            return redirect(url_for('laboratory.lab_reagent_inventory'))

        new_request = OtherOrder(
            item_id=item_id,
            quantity_requested=int(quantity),
            request_date=datetime.utcnow(),
            status="Pending",
            requested_by=current_user.id
        )

        db.session.add(new_request)
        db.session.commit()

        flash("Reagent restock request submitted!", "success")
        return redirect(url_for('laboratory.lab_reagent_inventory'))

    except Exception as e:
        flash(f"Error requesting reagent restock: {e}", "error")
        print(f"Debug: Error in laboratory.request_reagent_restock: {e}")
        return redirect(url_for('laboratory.lab_reagent_inventory'))
#########
bp.route('/generate_lab_report/<int:result_id>')
@login_required
def generate_lab_report(result_id):
    """Generates a PDF lab test report."""
    if current_user.role not in ['laboratory', 'admin']:
        flash("You don't have permission to access this page.", "error")
        return redirect(url_for('home'))

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
def search_patient():
    """Search for a patient and fetch their lab history on the same page."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('laboratory.index'))

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
def abnormal_results():
    """Displays lab test results that are outside normal ranges."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

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
        flash(f'Error fetching abnormal results: {e}', 'error')
        print(f"Debug: Error in laboratory.abnormal_results: {e}")
        return redirect(url_for('laboratory.index'))
#dashboard
@bp.route('/dashboard')
@login_required
def dashboard():
    """Displays key lab statistics and trends."""
    if current_user.role not in ['laboratory', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    total_tests = LabResult.query.count()
    pending_tests = RequestedLab.query.filter_by(status=0).count()
    abnormal_tests = LabResult.query.filter(LabResult.result.ilike('%abnormal%')).count()

    return render_template('laboratory/dashboard.html',
                           total_tests=total_tests,
                           pending_tests=pending_tests,
                           abnormal_tests=abnormal_tests)


