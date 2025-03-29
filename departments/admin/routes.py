from flask import Blueprint, render_template, flash, redirect, url_for, request
from extensions import db 
from flask_login import login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SelectField, SubmitField
from wtforms.validators import DataRequired, Length
from werkzeug.security import generate_password_hash
from . import bp  # Import the blueprint
from departments.models.user import User  # Import User model
from departments.models.admin import Log  # Corrected to use Log model from log module
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define roles for the dropdown
ROLES = [
    ('admin', 'Admin'),
    ('records', 'Records'),
    ('nursing', 'Nursing'),
    ('pharmacy', 'Pharmacy'),
    ('stores', 'Stores')
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

@bp.route('/index', methods=['GET'])
@login_required
def index():
    """Admin dashboard showing all users."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/index by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/index by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

    try:
        users = User.query.order_by(User.username).all()
        logger.info(f"Admin {current_user.id} accessed dashboard")
        db.session.add(Log(
            level='INFO',
            message=f"Admin {current_user.username} (ID: {current_user.id}) accessed dashboard",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return render_template('admin/index.html', users=users)
    except Exception as e:
        flash(f'Error loading dashboard: {e}', 'error')
        logger.error(f"Error in admin.index: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading dashboard: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))
@bp.route('/add_user', methods=['GET', 'POST'])
@login_required
def add_user():
    """Allows admin to add a new user."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/add_user by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/add_user by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

    form = AddUserForm()
    if form.validate_on_submit():
        try:
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
            flash(f'Error adding user: {e}', 'error')
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
def manage_users():
    """Admin page to manage existing users."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/manage_users by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/manage_users by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

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
        flash(f'Error loading users: {e}', 'error')
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
def edit_user(user_id):
    """Admin page to edit an existing user."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/edit_user/{user_id} by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/edit_user/{user_id} by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

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
            flash(f'Error updating user: {e}', 'error')
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
def delete_user(user_id):
    """Admin action to delete a user."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/delete_user/{user_id} by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/delete_user/{user_id} by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

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
        flash(f'Error deleting user: {e}', 'error')
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
def system_overview():
    """Admin page showing system stats."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/system_overview by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/system_overview by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

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
        flash(f'Error loading overview: {e}', 'error')
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
def logs():
    """Admin page showing system logs."""
    if current_user.role != 'admin':
        flash('Unauthorized access. Admin only.', 'error')
        logger.warning(f"Unauthorized access attempt to /admin/logs by user {current_user.id}")
        db.session.add(Log(
            level='WARNING',
            message=f"Unauthorized access attempt to /admin/logs by user {current_user.id}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('login'))

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
        flash(f'Error loading logs: {e}', 'error')
        logger.error(f"Error in admin.logs: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading logs: {str(e)}",
            user_id=current_user.id,
            source='admin'
        ))
        db.session.commit()
        return redirect(url_for('admin.index'))