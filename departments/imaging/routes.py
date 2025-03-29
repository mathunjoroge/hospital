from flask import  render_template, redirect, url_for, request, flash
from flask import current_app  # Import current_app
from flask_login import login_required, current_user
from extensions import db
from flask import session
from datetime import datetime
import uuid
from uuid import uuid4
import os
from sqlalchemy.orm import relationship, joinedload
from . import bp  # Import the blueprint
from flask_socketio import SocketIO
from departments.models.records import PatientWaitingList,Patient
from departments.models.medicine import (RequestedImage,Imaging,ImagingResult
)

socketio = SocketIO()
 # index
@bp.route('/', methods=['GET'])
@login_required
def index():
    """Displays the imaging waiting list with pending imaging requests."""
    if current_user.role not in ['imaging', 'admin']: 
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch all pending imaging requests (status=0)
        pending_requests = RequestedImage.query.filter_by(status=0).options(
            joinedload(RequestedImage.patient),
            joinedload(RequestedImage.imaging)  # Eager load imaging details
        ).all()

        # If no pending requests exist, inform the user
        if not pending_requests:
            flash('No pending imaging requests at the moment.', 'info')

        return render_template(
            'imaging/index.html',
            pending_requests=pending_requests
        )

    except Exception as e:
        flash(f'Error fetching pending imaging requests: {e}', 'error')
        print(f"Debug: Error in imaging.index: {e}")  # Debugging
        return redirect(url_for('home'))
    
@bp.route('/process_imaging_request/<int:request_id>', methods=['GET', 'POST'])
@login_required
def process_imaging_request(request_id):
    """Handles processing an imaging request and uploading DICOM images."""
    if current_user.role not in ['imaging', 'admin']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the requested imaging by ID
        imaging_request = RequestedImage.query.get_or_404(request_id)

        # Fetch the associated imaging details
        imaging = Imaging.query.get_or_404(imaging_request.imaging_id)

        if request.method == 'POST':
            # Generate a unique result_id
            result_id = str(uuid.uuid4())

            # Handle uploaded image file
            uploaded_file = request.files.get('image_file')
            if not uploaded_file or not uploaded_file.filename.endswith('.dcm'):
                flash('Invalid DICOM file uploaded!', 'error')
                return render_template(
                    'imaging/process.html',
                    imaging_request=imaging_request,
                    imaging=imaging
                )

            # Save the DICOM file to the upload folder
            dicom_upload_folder = current_app.config['DICOM_UPLOAD_FOLDER']
            file_name = f"{result_id}.dcm"  # Use result_id as the file name
            file_path = os.path.join(dicom_upload_folder, file_name)
            uploaded_file.save(file_path)

            # Create or update the imaging result record
            imaging_result = ImagingResult(
                result_id=result_id,
                patient_id=imaging_request.patient_id,
                imaging_id=imaging.id,
                test_date=datetime.utcnow(),
                result_notes=request.form.get('result_notes', ''),
                updated_by=current_user.id,
                dicom_file_path=file_path  # Store the file path
            )
            db.session.add(imaging_result)
            db.session.commit()

            # Update the imaging request status to processed (e.g., status=1)
            imaging_request.status = 1
            imaging_request.result_id = result_id  # Link the request to the result via result_id
            db.session.commit()

            flash('Imaging results submitted successfully!', 'success')
            return redirect(url_for('imaging.index'))  # Redirect back to the imaging index page

        # Render the form on GET request
        return render_template(
            'imaging/process.html',
            imaging_request=imaging_request,
            imaging=imaging
        )

    except Exception as e:
        flash(f'Error processing imaging request: {e}', 'error')
        db.session.rollback()  # Rollback changes in case of error
        print(f"Debug: Error in imaging.process_imaging_request: {e}")  # Debugging
        return redirect(url_for('imaging.index'))
from flask import send_from_directory

@bp.route('/view_imaging_results/<string:result_id>', methods=['GET'])
@login_required
def view_imaging_results(result_id):
    """Displays imaging results including DICOM images."""
    if current_user.role not in ['medicine', 'imaging']:
        flash('You do not have permission to access this page.', 'error')
        return redirect(url_for('home'))

    try:
        # Fetch the imaging result by result_id
        imaging_result = ImagingResult.query.filter_by(result_id=result_id).options(
            joinedload(ImagingResult.imaging)
        ).first()

        if not imaging_result:
            flash(f'Imaging result with ID {result_id} not found!', 'error')
            return redirect(url_for('imaging.index'))

        # Serve the DICOM file if available
        if imaging_result.dicom_file_path:
            # Extract the file name from the path
            file_name = os.path.basename(imaging_result.dicom_file_path)
            return send_from_directory(
                directory=current_app.config['DICOM_UPLOAD_FOLDER'],
                path=file_name,
                mimetype='application/dicom',
                as_attachment=False  # Display the file in the browser if possible
            )

        flash('No image available for this result.', 'info')
        return render_template(
            'imaging/view.html',
            imaging_result=imaging_result,
            imaging=imaging_result.imaging
        )

    except Exception as e:
        flash(f'Error fetching imaging results: {e}', 'error')
        print(f"Debug: Error in imaging.view_imaging_results: {e}")  # Debugging
        return redirect(url_for('imaging.index'))