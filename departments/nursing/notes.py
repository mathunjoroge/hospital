import logging
from datetime import datetime, timedelta

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from departments.models.admin import Log
from departments.models.nursing import (
    Messages,
    Notifications,
    NursingCareTask,
    NursingNote,
    Partogram,
)
from departments.models.records import Patient
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@bp.route('/view_notes', methods=['GET'])
@login_required
@roles_required('nursing', 'medicine', 'admin')
def view_notes():
    """View all nursing notes for patients."""

    try:
        notes = NursingNote.query.order_by(NursingNote.timestamp.desc()).all()
        logger.info(f"Nurse {current_user.id} viewed nursing notes")
        db.session.add(Log(
            level='INFO',
            message=f"Nurse {current_user.username} (ID: {current_user.id}) viewed nursing notes",
            user_id=current_user.id,
            source='nursing'
        ))
        db.session.commit()
        return render_template('nursing/view_notes.html', notes=notes)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in nursing.view_notes: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading nursing notes: {str(e)}",
            user_id=current_user.id,
            source='nursing'
        ))
        db.session.commit()
        return redirect(url_for('home'))

@bp.route('/add_note', methods=['GET', 'POST'])
@login_required
@roles_required('nursing', 'admin')
def add_note():
    """Add a new nursing note for a patient with Kardex details."""

    if request.method == 'POST':
        try:
            patient_id = request.form['patient_id']
            note = request.form['note']
            allergies = request.form.get('allergies', '')
            code_status = request.form.get('code_status', '')
            medications = request.form.get('medications', '')  # Could be a delimited string or JSON
            shift_update = request.form.get('shift_update', '')

            # Combine into note if not storing separately

            new_note = NursingNote(
                patient_id=patient_id,
                nurse_id=current_user.id,
                note=note,
                #Uncomment if using separate fields:
                allergies=allergies,
                code_status=code_status,
                medications=medications,
                shift_update=shift_update
            )
            db.session.add(new_note)
            db.session.commit()
            logger.info(f"Nurse {current_user.id} added note for patient {patient_id}")
            db.session.add(Log(
                level='INFO',
                message=f"Nurse {current_user.username} (ID: {current_user.id}) added note for patient {patient_id}",
                user_id=current_user.id,
                source='nursing'
            ))
            db.session.commit()
            flash('Note added successfully.', 'success')
            return redirect(url_for('nursing.view_notes'))
        except Exception as e:
            db.session.rollback()
            flash('Something went wrong. Please try again.', 'error')
            logger.error(f"Error in nursing.add_note: {e}", exc_info=True)
            db.session.add(Log(
                level='ERROR',
                message=f"Error adding note: {str(e)}",
                user_id=current_user.id,
                source='nursing'
            ))
            db.session.commit()
            return render_template('nursing/add_note.html', patients=Patient.query.all())

    try:
        patients = Patient.query.order_by(Patient.name).all()
        return render_template('nursing/add_note.html', patients=patients)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in nursing.add_note: {e}", exc_info=True)
        return render_template('nursing/add_note.html', patients=[])


@bp.route('/patient/<string:patient_id>')
@login_required
@roles_required('nursing', 'medicine', 'admin')
def patient_dashboard(patient_id):

    try:
        partogram = Partogram.query.filter_by(patient_id=patient_id).order_by(Partogram.timestamp.desc()).first()
        notes = NursingNote.query.filter_by(patient_id=patient_id).order_by(NursingNote.timestamp.desc()).limit(5).all()
        tasks = NursingCareTask.query.filter_by(patient_id=patient_id, status='Pending').order_by(NursingCareTask.created_at.desc()).all()
        return render_template('nursing/patient_dashboard.html', patient_id=patient_id, partogram=partogram, notes=notes, tasks=tasks)
    except Exception as e:
        flash(f'Error fetching patient dashboard data: {str(e)}', 'error')
        return redirect(url_for('nursing.index'))

@bp.route('/patient/<string:patient_id>/add-note', methods=['GET', 'POST'])
@login_required
@roles_required('nursing', 'admin')
def add_patient_note(patient_id):
    # Logic for adding note would go here
    return "Note added"


@bp.route('/shift_handover')
@login_required
@roles_required('nursing', 'admin', 'medicine', 'doctor')
def shift_handover():

    try:
        recent_time = datetime.utcnow() - timedelta(hours=12)
        patients = db.session.query(Partogram.patient_id).filter(Partogram.timestamp > recent_time).distinct().all()

        handover_data = []
        for (patient_id,) in patients:
            partogram = Partogram.query.filter_by(patient_id=patient_id).order_by(Partogram.timestamp.desc()).first()
            note = NursingNote.query.filter_by(patient_id=patient_id).order_by(NursingNote.timestamp.desc()).first()
            tasks = NursingCareTask.query.filter_by(patient_id=patient_id, status='Pending').order_by(NursingCareTask.created_at.desc()).all()

            handover_data.append({'patient_id': patient_id, 'partogram': partogram, 'note': note, 'tasks': tasks})

        return render_template('nursing/shift_handover.html', handover_data=handover_data)
    except Exception as e:
        flash(f'Error fetching shift handover data: {str(e)}', 'error')
        return redirect(url_for('nursing.index'))

@bp.route('/communicate-doctor', methods=['GET', 'POST'])
@bp.route('/patient/<string:patient_id>/communicate-doctor', methods=['GET', 'POST'])
@login_required
@roles_required('nursing', 'admin')
def communicate_doctor(patient_id=None):
    if request.method == 'POST':
        try:
            message_text = request.form.get('message')
            doctor_id = request.form.get('doctor_id')
            p_id = request.form.get('patient_id', patient_id)
            if not message_text or not doctor_id:
                flash('Message and doctor selection are required.', 'error')
                return redirect(url_for('nursing.communicate_doctor', patient_id=p_id) if p_id else url_for('nursing.communicate_doctor'))

            new_message = Messages(
                sender_id=current_user.id,
                receiver_id=int(doctor_id),
                patient_id=p_id or '',
                message=message_text,
                timestamp=datetime.utcnow()
            )
            db.session.add(new_message)
            db.session.commit()
            flash('Message sent to doctor.', 'success')
            return redirect(url_for('nursing.communicate_doctor', patient_id=p_id) if p_id else url_for('nursing.communicate_doctor'))
        except ValueError:
            flash('Invalid doctor selection.', 'error')
            return redirect(url_for('nursing.communicate_doctor', patient_id=patient_id) if patient_id else url_for('nursing.communicate_doctor'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error sending message: {str(e)}', 'error')
            return redirect(url_for('nursing.communicate_doctor', patient_id=patient_id) if patient_id else url_for('nursing.communicate_doctor'))

    try:
        doctors = User.query.filter(User.role.in_(['doctor', 'medicine'])).all()
        if not doctors:
            doctors = User.query.filter(User.role.in_(['doctor', 'medicine', 'admin'])).all()
        return render_template('nursing/communicate_doctor.html', doctors=doctors, patient_id=patient_id)
    except Exception as e:
        logger.error(f"Error fetching doctors in communicate_doctor: {e}", exc_info=True)
        flash(f'Error fetching doctors: {str(e)}', 'error')
        return redirect(url_for('nursing.index'))

@bp.route('/notifications', endpoint='notifications')
@bp.route('/get-notifications', endpoint='get_notifications')
@login_required
@roles_required('nursing', 'admin', 'medicine')
def get_notifications():
    try:
        notifications = Notifications.query.filter_by(receiver_id=current_user.id, is_read=False).order_by(Notifications.timestamp.desc()).all()
        return render_template('nursing/notifications.html', notifications=notifications)
    except Exception as e:
        flash(f'Error fetching notifications: {str(e)}', 'error')
        return redirect(url_for('nursing.index'))

@bp.route('/mark_notification_read/<int:notification_id>', methods=['GET'])
@login_required
@roles_required('nursing')
def mark_notification_read(notification_id):

    try:
        notification = Notifications.query.filter_by(id=notification_id, receiver_id=current_user.id).first()
        if not notification:
            flash('Notification not found or you do not have permission to mark it as read.', 'error')
            return redirect(url_for('nursing.notifications'))

        notification.is_read = True
        db.session.commit()
        flash('Notification marked as read.', 'success')
        return redirect(url_for('nursing.notifications'))
    except Exception as e:
        db.session.rollback()
        flash(f'Error marking notification as read: {str(e)}', 'error')
        return redirect(url_for('nursing.notifications'))

@bp.route('/get-messages/<string:patient_id>', methods=['GET'])
@login_required
@roles_required('nursing')
def get_messages(patient_id):
    pass

