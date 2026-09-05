import base64
import io
import logging
from datetime import datetime

import pyotp
import qrcode
from flask import flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from werkzeug.security import generate_password_hash
from wtforms import PasswordField, SelectField, StringField, SubmitField
from wtforms.validators import DataRequired, Length

from departments.models.admin import Log
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define roles for the dropdown
ROLES = [
    ('admin', 'Admin'),
    ('records', 'Records'),
    ('nursing', 'Nursing'),
    ('pharmacy', 'Pharmacy'),
    ('stores', 'Stores'),
    ('mortuary', 'Mortuary')
]

class AddUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=80)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6, max=120)])
    role = SelectField('Role', choices=ROLES, validators=[DataRequired()])
    submit = SubmitField('Add User')
    #edit user
class EditUserForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=80)])
    role = SelectField('Role', choices=ROLES, validators=[DataRequired()])
    submit = SubmitField('Update User')


@bp.route('/admin/switch_user', methods=['POST'])
@bp.route('/switch_user', methods=['POST'])
@login_required
@roles_required('admin')
def switch_user():

    new_role = request.form.get('role')
    allowed_roles = ['records', 'billing', 'nursing', 'laboratory', 'imaging', 'pharmacy', 'medicine', 'stores', 'hr', 'mortuary', 'admin']

    if new_role not in allowed_roles:
        flash('Invalid role selected.', 'danger')
        return redirect(url_for('home'))

    session['switched_user'] = new_role
    flash(f'Switched to {new_role} role.', 'success')

    # Role-to-homepage mapping
    role_homepages = {
        'records': 'records.index',
        'billing': 'billing.index',
        'nursing': 'nursing.index',
        'laboratory': 'laboratory.index',
        'imaging': 'imaging.index',
        'pharmacy': 'pharmacy.index',
        'medicine': 'medicine.index',
        'stores': 'stores.index',
        'hr': 'hr.index',
        'mortuary': 'mortuary.index',
        'admin': 'admin.index'
    }

    # Redirect to the role-specific homepage
    return redirect(url_for(role_homepages.get(new_role, 'home')))

@bp.route('/admin/revert_user')
@login_required
@roles_required('admin')
def revert_user():

    session.pop('switched_user', None)
    flash('Reverted to admin role.', 'success')
    return redirect(url_for('admin.index'))

def get_effective_role():
    if current_user.is_authenticated and current_user.role == 'admin' and 'switched_user' in session:
        return session['switched_user']
    return current_user.role if current_user.is_authenticated else None

@bp.route('/index', methods=['GET'])
@bp.route('/', methods=['GET'])  # Add this to handle /admin directly
@login_required
@roles_required('admin')
def index():
    """Admin dashboard showing all users."""

    try:
        department = request.args.get('department', 'admin')  # Get department from query parameter
        users = User.query.order_by(User.username).all()
        logger.info(f"Admin {current_user.id} accessed dashboard with department {department}")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) accessed dashboard with department {department}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return render_template('admin/index.html', users=users, department=department)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in admin.index: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading dashboard: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))


def validate_password_complexity(password):
    """Validate password strength based on security requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(char.isupper() for char in password):
        return False, "Password must contain at least one uppercase letter."
    if not any(char.isdigit() for char in password):
        return False, "Password must contain at least one digit."
    if not any(char in "!@#$%^&*()-_+=[]{}|;:,.<>?" for char in password):
        return False, "Password must contain at least one special character."
    return True, "Password is strong."


@bp.route('/add_user', methods=['GET', 'POST'])
@login_required
@roles_required('admin')
def add_user():
    """Allows admin to add a new user."""

    form = AddUserForm()
    if form.validate_on_submit():
        try:
            valid_pwd, pwd_msg = validate_password_complexity(form.password.data)
            if not valid_pwd:
                flash(pwd_msg, 'error')
                return render_template('admin/add_user.html', form=form)

            if User.query.filter_by(username=form.username.data).first():
                flash('Username already exists.', 'error')
                logger.warning(f"Duplicate username attempt: {form.username.data} by admin {current_user.id}")
                db.session.add(Log(
                    level='WARNING',
                    message=f"Duplicate username attempt: {form.username.data} by admin {current_user.id}",
                    user_id=current_user.id,
                    source='admin'
                ))
                db.session.commit()
                return render_template('admin/add_user.html', form=form)

            new_user = User(
                username=form.username.data,
                password=generate_password_hash(form.password.data, method='pbkdf2:sha256'),
                role=form.role.data
            )

            db.session.add(new_user)
            db.session.commit()
            logger.info(f"Admin {current_user.id} added user {new_user.username}")
            db.session.add(Log(
                level='INFO',
                message=f"User {new_user.username} (ID: {new_user.id}) created with role {new_user.role} by admin {current_user.id}",
                user_id=current_user.id,
                source='admin'
            ))
            db.session.commit()
            flash(f'User {form.username.data} added successfully.', 'success')
            return redirect(url_for('admin.index'))

        except Exception as e:
            db.session.rollback()
            flash('Something went wrong. Please try again.', 'error')
            logger.error(f"Error in admin.add_user: {e}", exc_info=True)
            db.session.add(Log(
                level='ERROR',
                message=f"Error adding user: {str(e)}",
                user_id=current_user.id,
                source='admin'
            ))
            db.session.commit()
            return render_template('admin/add_user.html', form=form)

    return render_template('admin/add_user.html', form=form)

@bp.route('/manage_users', methods=['GET'])
@login_required
@roles_required('admin')
def manage_users():
    """Admin page to manage existing users."""

    try:
        users = User.query.order_by(User.username).all()
        logger.info(f"Admin {current_user.id} accessed manage users page")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) accessed manage users page",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return render_template('admin/manage_users.html', users=users)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in admin.manage_users: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading manage users page: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))
@bp.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
@roles_required('admin')
def edit_user(user_id):
    """Admin page to edit an existing user."""

    user = User.query.get_or_404(user_id)
    form = EditUserForm(obj=user)  # Prepopulate form with user data

    if form.validate_on_submit():
        try:
            # Check for username conflicts (excluding the current user)
            existing_user = User.query.filter_by(username=form.username.data).first()
            if existing_user and existing_user.id != user.id:
                flash('Username already exists.', 'error')
                logger.warning(f"Duplicate username attempt: {form.username.data} by admin {current_user.id} for user {user.id}")
                db.session.add(Log(
                    level='WARNING',
                    message=f"Duplicate username attempt: {form.username.data} by admin {current_user.id} for user {user.id}",
                    user_id=current_user.id,
                    source='admin'
                ))
                db.session.commit()
                return render_template('admin/edit_user.html', form=form, user=user)

            user.username = form.username.data
            user.role = form.role.data
            db.session.commit()
            logger.info(f"Admin {current_user.id} updated user {user.username} (ID: {user.id})")
            db.session.add(Log(
                level='INFO',
                message=f"Admin {current_user.username} (ID: {current_user.id}) updated user {user.username} (ID: {user.id}) to role {user.role}",
                user_id=current_user.id,
                source='admin'
            ))
            db.session.commit()
            flash(f'User {user.username} updated successfully.', 'success')
            return redirect(url_for('admin.manage_users'))
        except Exception as e:
            db.session.rollback()
            flash('Something went wrong. Please try again.', 'error')
            logger.error(f"Error in admin.edit_user: {e}", exc_info=True)
            db.session.add(Log(
                level='ERROR',
                message=f"Error updating user {user.id}: {str(e)}",
                user_id=current_user.id,
                source='admin'
            ))
            db.session.commit()
            return render_template('admin/edit_user.html', form=form, user=user)

    return render_template('admin/edit_user.html', form=form, user=user)

@bp.route('/delete_user/<int:user_id>', methods=['POST'])
@login_required
@roles_required('admin')
def delete_user(user_id):
    """Admin action to delete a user."""

    try:
        user = User.query.get_or_404(user_id)
        if user.id == current_user.id:
            flash('You cannot delete your own account.', 'error')
            logger.warning(f"Admin {current_user.id} attempted to delete own account")
            db.session.add(Log(
                level='WARNING',
                message=f"Admin {current_user.username} (ID: {current_user.id}) attempted to delete own account",
                user_id=current_user.id,
                source='admin'
            ))
            db.session.commit()
            return redirect(url_for('admin.manage_users'))

        db.session.delete(user)
        db.session.commit()
        logger.info(f"Admin {current_user.id} deleted user {user.username} (ID: {user.id})")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) deleted user {user.username} (ID: {user.id})",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        flash(f'User {user.username} deleted successfully.', 'success')
        return redirect(url_for('admin.manage_users'))
    except Exception as e:
        db.session.rollback()
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in admin.delete_user: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error deleting user {user_id}: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('admin.manage_users'))


@bp.route('/system_overview', methods=['GET'])
@login_required
@roles_required('admin')
def system_overview():
    """Admin page showing system stats."""

    try:
        user_count = User.query.count()
        logger.info(f"Admin {current_user.id} accessed system overview")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) accessed system overview",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return render_template('admin/system_overview.html', user_count=user_count)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in admin.system_overview: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading system overview: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('home'))

@bp.route('/logs', methods=['GET'])
@login_required
@roles_required('admin')
def logs():
    """Admin page showing system logs."""

    try:
        logs = Log.query.order_by(Log.timestamp.desc()).limit(100).all()  # Last 100 logs
        logger.info(f"Admin {current_user.id} viewed system logs")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) viewed system logs",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return render_template('admin/logs.html', logs=logs)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in admin.logs: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading logs: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('admin.index'))

@bp.route('/mfa/setup', methods=['GET', 'POST'])
@login_required
@roles_required('admin')
def mfa_setup():
    user = current_user
    secret = session.get('mfa_setup_secret')
    if not secret:
        secret = pyotp.random_base32()
        session['mfa_setup_secret'] = secret

    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(name=user.username, issuer_name="HMIS Hospital")

    img = qrcode.make(provisioning_uri)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    qr_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    if request.method == 'POST':
        code = request.form.get('code', '').strip()
        if totp.verify(code):
            user.totp_secret = secret
            user.mfa_enabled = True
            db.session.commit()
            session.pop('mfa_setup_secret', None)
            flash('MFA has been successfully enabled for your account!', 'success')
            return redirect(url_for('admin.index'))
        else:
            flash('Invalid MFA verification code. Please try again.', 'error')

    return render_template('admin/mfa_setup.html', secret=secret, qr_b64=qr_b64)


@bp.route('/admin/audit-trail', methods=['GET'])
@bp.route('/audit-trail', methods=['GET'])
@login_required
@roles_required('admin')
def audit_trail():
    """Admin dashboard view for persistent system audit logs."""
    from departments.models.compliance import AuditLog

    page = request.args.get('page', 1, type=int)
    action_filter = request.args.get('action', '').strip()
    username_filter = request.args.get('username', '').strip()
    resource_type_filter = request.args.get('resource_type', '').strip()

    query = AuditLog.query

    if action_filter:
        query = query.filter(AuditLog.action.ilike(f"%{action_filter}%"))
    if username_filter:
        query = query.filter(AuditLog.username.ilike(f"%{username_filter}%"))
    if resource_type_filter:
        query = query.filter(AuditLog.resource_type.ilike(f"%{resource_type_filter}%"))

    pagination = query.order_by(AuditLog.timestamp.desc()).paginate(page=page, per_page=30, error_out=False)

    if request.args.get('format') == 'json':
        return {
            "total": pagination.total,
            "page": page,
            "pages": pagination.pages,
            "logs": [log.to_dict() for log in pagination.items]
        }

    return render_template('admin/audit_trail.html', pagination=pagination, logs=pagination.items)


@bp.route('/audit-trail/export', methods=['GET'])
@login_required
@roles_required('admin')
def export_audit_trail():

    """Export system audit logs as structured JSON for SIEM integration."""
    from flask import jsonify

    from departments.models.compliance import AuditLog

    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(1000).all()
    return jsonify({
        "system": "HMIS",
        "exported_at": datetime.now().isoformat(),
        "count": len(logs),
        "audit_logs": [log_item.to_dict() for log_item in logs]
    })


@bp.route('/admin/outbound-notifications', methods=['GET'])
@bp.route('/outbound-notifications', methods=['GET'])
@login_required
@roles_required('admin')
def outbound_notifications():
    """Admin dashboard view for outbound patient notification logs."""
    from flask import jsonify

    from departments.models.notification_log import OutboundNotificationLog

    page = request.args.get('page', 1, type=int)
    event_filter = request.args.get('event_type', '').strip()
    status_filter = request.args.get('status', '').strip()
    recipient_filter = request.args.get('recipient', '').strip()

    query = OutboundNotificationLog.query

    if event_filter:
        query = query.filter(OutboundNotificationLog.event_type.ilike(f"%{event_filter}%"))
    if status_filter:
        query = query.filter(OutboundNotificationLog.status == status_filter.upper())
    if recipient_filter:
        query = query.filter(OutboundNotificationLog.recipient.ilike(f"%{recipient_filter}%"))

    pagination = query.order_by(OutboundNotificationLog.created_at.desc()).paginate(page=page, per_page=30, error_out=False)

    if request.args.get('format') == 'json':
        return jsonify({
            "total": pagination.total,
            "page": page,
            "pages": pagination.pages,
            "logs": [
                {
                    "id": log.id,
                    "patient_id": log.patient_id,
                    "recipient": log.recipient,
                    "channel": log.channel,
                    "event_type": log.event_type,
                    "subject": log.subject,
                    "status": log.status,
                    "error_message": log.error_message,
                    "sent_at": log.sent_at.isoformat() if log.sent_at else None,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                }
                for log in pagination.items
            ]
        })

    return render_template('admin/logs.html', logs=pagination.items)
