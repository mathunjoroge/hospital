import base64
import io
import logging
from datetime import datetime, timedelta, timezone

import pyotp
import qrcode
from flask import (
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from werkzeug.security import generate_password_hash
from wtforms import (
    BooleanField,
    EmailField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
)
from wtforms.validators import DataRequired, Email, Length, Optional

from departments.api.audit import log_audit_event
from departments.models.admin import Log
from departments.models.user import User
from departments.rbac import roles_required
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define roles for the dropdown
ROLES = [
    ("admin", "Admin"),
    ("records", "Records"),
    ("billing", "Billing"),
    ("nursing", "Nursing"),
    ("laboratory", "Laboratory"),
    ("imaging", "Imaging"),
    ("pharmacy", "Pharmacy"),
    ("medicine", "Medicine"),
    ("stores", "Stores"),
    ("hr", "HR"),
    ("mortuary", "Mortuary"),
    ("theatre", "Theatre / Surgery"),
    ("oncology", "Oncology"),
    ("mch", "MCH / ANC / NICU"),
    ("icu", "ICU / HDU"),
    ("renal", "Renal / Dialysis"),
]


class AddUserForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=4, max=80)]
    )
    full_name = StringField("Full Name", validators=[Optional(), Length(max=120)])
    email = EmailField(
        "Email Address", validators=[Optional(), Email(), Length(max=120)]
    )
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=120)]
    )
    role = SelectField("Role", choices=ROLES, validators=[DataRequired()])
    submit = SubmitField("Add User")


class EditUserForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=4, max=80)]
    )
    full_name = StringField("Full Name", validators=[Optional(), Length(max=120)])
    email = EmailField(
        "Email Address", validators=[Optional(), Email(), Length(max=120)]
    )
    role = SelectField("Role", choices=ROLES, validators=[DataRequired()])
    is_active = BooleanField("Account Active", default=True)
    submit = SubmitField("Update User")


class ResetPasswordForm(FlaskForm):
    new_password = PasswordField(
        "New Password", validators=[DataRequired(), Length(min=8, max=120)]
    )
    submit = SubmitField("Reset Password")


@bp.route("/admin/switch_user", methods=["POST"])
@bp.route("/switch_user", methods=["POST"])
@login_required
def switch_user():
    # Guard against non-admins using the real role, NOT the effective role.
    if getattr(current_user, "role", None) != "admin":
        abort(403)

    new_role = request.form.get("role")
    allowed_roles = [
        "records",
        "billing",
        "nursing",
        "laboratory",
        "imaging",
        "pharmacy",
        "medicine",
        "stores",
        "hr",
        "mortuary",
        "theatre",
        "oncology",
        "mch",
        "icu",
        "renal",
        "admin",
    ]

    if new_role not in allowed_roles:
        flash("Invalid role selected.", "danger")
        return redirect(url_for("home"))

    session["switched_user"] = new_role
    log_audit_event(
        action="ROLE_SWITCH",
        resource_type="UserSession",
        resource_id=str(current_user.id),
        details={"switched_to_role": new_role, "original_role": current_user.role},
    )
    flash(f"Switched to {new_role} role.", "success")

    # Role-to-homepage mapping
    role_homepages = {
        "records": "records.index",
        "billing": "billing.index",
        "nursing": "nursing.view_notes",
        "laboratory": "laboratory.lab_tests",
        "imaging": "imaging.index",
        "pharmacy": "pharmacy.index",
        "medicine": "medicine.index",
        "stores": "stores.index",
        "hr": "hr.index",
        "mortuary": "mortuary.index",
        "theatre": "theatre.get_or_dashboard_ui",
        "oncology": "medicine.oncology",
        "mch": "mch.index",
        "icu": "icu.icu_flowsheet",
        "renal": "renal.list_sessions",
        "admin": "admin.index",
    }

    return redirect(url_for(role_homepages.get(new_role, "home")))


@bp.route("/admin/revert_user")
@bp.route("/revert_user")
@login_required
def revert_user():
    if getattr(current_user, "role", None) == "admin":
        prev = session.pop("switched_user", None)
        log_audit_event(
            action="ROLE_REVERT",
            resource_type="UserSession",
            resource_id=str(current_user.id),
            details={"reverted_from_role": prev, "restored_role": "admin"},
        )
        flash("Reverted to admin role.", "success")
        return redirect(url_for("admin.index"))
    abort(403)


def get_effective_role():
    if (
        current_user.is_authenticated
        and current_user.role == "admin"
        and "switched_user" in session
    ):
        return session["switched_user"]
    return current_user.role if current_user.is_authenticated else None


@bp.route("/index", methods=["GET"])
@bp.route("/", methods=["GET"])
@login_required
@roles_required("admin")
def index():
    """Admin dashboard showing all users."""

    try:
        department = request.args.get("department", "admin")
        users = User.query.order_by(User.username).all()
        now = datetime.now(timezone.utc)

        # Quick stats for the dashboard
        locked_count = sum(1 for u in users if u.locked_until and u.locked_until > now)
        mfa_count = sum(1 for u in users if u.mfa_enabled)
        inactive_count = sum(1 for u in users if not getattr(u, "is_active", True))

        # Expiring credentials count (within 30 days)
        expiring_creds_count = 0
        try:
            from departments.models.hr import StaffCredential

            cutoff = now.date() + timedelta(days=30)
            expiring_creds_count = StaffCredential.query.filter(
                StaffCredential.expiry_date <= cutoff
            ).count()
        except Exception:
            pass

        logger.info(
            f"Admin {current_user.id} accessed dashboard with department {department}"
        )
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} (ID: {current_user.id}) accessed dashboard with department {department}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return render_template(
            "admin/index.html",
            users=users,
            department=department,
            locked_count=locked_count,
            mfa_count=mfa_count,
            inactive_count=inactive_count,
            expiring_creds_count=expiring_creds_count,
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in admin.index: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading dashboard: {e!s}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return redirect(url_for("login"))


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


@bp.route("/add_user", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def add_user():
    """Allows admin to add a new user."""

    form = AddUserForm()
    if form.validate_on_submit():
        try:
            valid_pwd, pwd_msg = validate_password_complexity(form.password.data)
            if not valid_pwd:
                flash(pwd_msg, "error")
                return render_template("admin/add_user.html", form=form)

            if User.query.filter_by(username=form.username.data).first():
                flash("Username already exists.", "error")
                logger.warning(
                    f"Duplicate username attempt: {form.username.data} by admin {current_user.id}"
                )
                db.session.add(
                    Log(
                        level="WARNING",
                        message=f"Duplicate username attempt: {form.username.data} by admin {current_user.id}",
                        user_id=current_user.id,
                        source="admin",
                    )
                )
                db.session.commit()
                return render_template("admin/add_user.html", form=form)

            # Check email uniqueness if provided
            email_val = form.email.data.strip() if form.email.data else None
            if email_val and User.query.filter_by(email=email_val).first():
                flash("Email address already in use by another account.", "error")
                return render_template("admin/add_user.html", form=form)

            new_user = User(
                username=form.username.data,
                full_name=form.full_name.data.strip() if form.full_name.data else None,
                email=email_val,
                password=generate_password_hash(
                    form.password.data, method="pbkdf2:sha256"
                ),
                role=form.role.data,
                is_active=True,
            )

            db.session.add(new_user)
            db.session.commit()
            logger.info(f"Admin {current_user.id} added user {new_user.username}")
            db.session.add(
                Log(
                    level="INFO",
                    message=f"User {new_user.username} (ID: {new_user.id}) created with role {new_user.role} by admin {current_user.id}",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            log_audit_event(
                action="USER_CREATED",
                resource_type="User",
                resource_id=str(new_user.id),
                details={
                    "username": new_user.username,
                    "role": new_user.role,
                    "email": new_user.email,
                },
            )
            db.session.commit()
            flash(f"User {form.username.data} added successfully.", "success")
            return redirect(url_for("admin.index"))

        except Exception as e:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.exception("Error in admin.add_user: ")
            db.session.add(
                Log(
                    level="ERROR",
                    message=f"Error adding user: {e!s}",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            db.session.commit()
            return render_template("admin/add_user.html", form=form)

    return render_template("admin/add_user.html", form=form)


@bp.route("/manage_users", methods=["GET"])
@login_required
@roles_required("admin")
def manage_users():
    """Admin page to manage existing users with search/filter/pagination support."""

    try:
        search_query = request.args.get("q", "").strip()
        role_filter = request.args.get("role", "").strip()
        status_filter = request.args.get("status", "").strip()
        page = request.args.get("page", 1, type=int)
        per_page = 25

        query = User.query

        if search_query:
            query = query.filter(
                db.or_(
                    User.username.ilike(f"%{search_query}%"),
                    User.email.ilike(f"%{search_query}%"),
                    User.full_name.ilike(f"%{search_query}%"),
                )
            )
        if role_filter:
            query = query.filter(User.role == role_filter)
        if status_filter == "locked":
            now = datetime.now(timezone.utc)
            query = query.filter(User.locked_until > now)
        elif status_filter == "inactive":
            query = query.filter(User.is_active == False)  # noqa: E712
        elif status_filter == "no_mfa":
            query = query.filter(User.mfa_enabled == False)  # noqa: E712

        pagination = query.order_by(User.username).paginate(
            page=page, per_page=per_page, error_out=False
        )
        now = datetime.now(timezone.utc)

        logger.info(f"Admin {current_user.id} accessed manage users page")
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} (ID: {current_user.id}) accessed manage users page",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return render_template(
            "admin/manage_users.html",
            users=pagination.items,
            pagination=pagination,
            roles=ROLES,
            search_query=search_query,
            role_filter=role_filter,
            status_filter=status_filter,
            now=now,
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in admin.manage_users: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading manage users page: {e!s}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return redirect(url_for("login"))


@bp.route("/edit_user/<int:user_id>", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def edit_user(user_id):
    """Admin page to edit an existing user."""

    user = User.query.get_or_404(user_id)
    form = EditUserForm(obj=user)

    if form.validate_on_submit():
        try:
            existing_user = User.query.filter_by(username=form.username.data).first()
            if existing_user and existing_user.id != user.id:
                flash("Username already exists.", "error")
                logger.warning(
                    f"Duplicate username attempt: {form.username.data} by admin {current_user.id} for user {user.id}"
                )
                db.session.add(
                    Log(
                        level="WARNING",
                        message=f"Duplicate username attempt: {form.username.data} by admin {current_user.id} for user {user.id}",
                        user_id=current_user.id,
                        source="admin",
                    )
                )
                db.session.commit()
                return render_template("admin/edit_user.html", form=form, user=user)

            # Check email uniqueness
            email_val = form.email.data.strip() if form.email.data else None
            if email_val:
                existing_email = User.query.filter_by(email=email_val).first()
                if existing_email and existing_email.id != user.id:
                    flash("Email address already in use by another account.", "error")
                    return render_template("admin/edit_user.html", form=form, user=user)

            old_role = user.role
            old_active = getattr(user, "is_active", True)
            user.username = form.username.data
            user.full_name = (
                form.full_name.data.strip() if form.full_name.data else user.full_name
            )
            user.email = email_val
            user.role = form.role.data
            user.is_active = form.is_active.data
            db.session.commit()
            logger.info(
                f"Admin {current_user.id} updated user {user.username} (ID: {user.id})"
            )
            db.session.add(
                Log(
                    level="INFO",
                    message=f"Admin {current_user.username} (ID: {current_user.id}) updated user {user.username} (ID: {user.id}) to role {user.role}",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            log_audit_event(
                action="USER_UPDATED",
                resource_type="User",
                resource_id=str(user.id),
                details={
                    "username": user.username,
                    "old_role": old_role,
                    "new_role": user.role,
                    "is_active": user.is_active,
                    "active_changed": old_active != user.is_active,
                },
            )
            db.session.commit()
            flash(f"User {user.username} updated successfully.", "success")
            return redirect(url_for("admin.manage_users"))
        except Exception as e:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.exception("Error in admin.edit_user: ")
            db.session.add(
                Log(
                    level="ERROR",
                    message=f"Error updating user {user.id}: {e!s}",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            db.session.commit()
            return render_template("admin/edit_user.html", form=form, user=user)

    return render_template("admin/edit_user.html", form=form, user=user)


@bp.route("/delete_user/<int:user_id>", methods=["POST"])
@login_required
@roles_required("admin")
def delete_user(user_id):
    """Admin action to delete a user."""

    try:
        user = User.query.get_or_404(user_id)
        if user.id == current_user.id:
            flash("You cannot delete your own account.", "error")
            logger.warning(f"Admin {current_user.id} attempted to delete own account")
            db.session.add(
                Log(
                    level="WARNING",
                    message=f"Admin {current_user.username} (ID: {current_user.id}) attempted to delete own account",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            db.session.commit()
            return redirect(url_for("admin.manage_users"))

        deleted_username = user.username
        db.session.delete(user)
        db.session.commit()
        logger.info(
            f"Admin {current_user.id} deleted user {deleted_username} (ID: {user_id})"
        )
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} (ID: {current_user.id}) deleted user {deleted_username} (ID: {user_id})",
                user_id=current_user.id,
                source="admin",
            )
        )
        log_audit_event(
            action="USER_DELETED",
            resource_type="User",
            resource_id=str(user_id),
            details={"username": deleted_username},
        )
        db.session.commit()
        flash(f"User {deleted_username} deleted successfully.", "success")
        return redirect(url_for("admin.manage_users"))
    except Exception as e:
        db.session.rollback()
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in admin.delete_user: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error deleting user {user_id}: {e!s}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return redirect(url_for("admin.manage_users"))


@bp.route("/reset_password/<int:user_id>", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def reset_password(user_id):
    """Admin action to reset a user's password."""
    user = User.query.get_or_404(user_id)
    form = ResetPasswordForm()

    if form.validate_on_submit():
        try:
            valid_pwd, pwd_msg = validate_password_complexity(form.new_password.data)
            if not valid_pwd:
                flash(pwd_msg, "error")
                return render_template(
                    "admin/reset_password.html", form=form, user=user
                )

            user.password = generate_password_hash(
                form.new_password.data, method="pbkdf2:sha256"
            )
            # Clear lockout on manual password reset
            user.failed_login_attempts = 0
            user.locked_until = None
            db.session.commit()

            db.session.add(
                Log(
                    level="INFO",
                    message=f"Admin {current_user.username} reset password for user {user.username} (ID: {user.id})",
                    user_id=current_user.id,
                    source="admin",
                )
            )
            log_audit_event(
                action="PASSWORD_RESET",
                resource_type="User",
                resource_id=str(user.id),
                details={"target_username": user.username, "lockout_cleared": True},
            )
            db.session.commit()
            flash(
                f"Password for {user.username} has been reset successfully.", "success"
            )
            return redirect(url_for("admin.manage_users"))
        except Exception:
            db.session.rollback()
            flash("Something went wrong. Please try again.", "error")
            logger.exception("Error in admin.reset_password: ")
            return render_template("admin/reset_password.html", form=form, user=user)

    return render_template("admin/reset_password.html", form=form, user=user)


@bp.route("/lock_user/<int:user_id>", methods=["POST"])
@login_required
@roles_required("admin")
def lock_user(user_id):
    """Admin action to lock a user account (sets locked_until far in the future)."""
    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:
        flash("You cannot lock your own account.", "error")
        return redirect(url_for("admin.manage_users"))

    try:
        # Lock indefinitely (10 years)
        user.locked_until = datetime.now(timezone.utc) + timedelta(days=3650)
        db.session.commit()
        db.session.add(
            Log(
                level="WARNING",
                message=f"Admin {current_user.username} locked account for user {user.username} (ID: {user.id})",
                user_id=current_user.id,
                source="admin",
            )
        )
        log_audit_event(
            action="USER_LOCKED",
            resource_type="User",
            resource_id=str(user.id),
            details={"username": user.username, "locked_by": current_user.username},
        )
        db.session.commit()
        flash(f"Account for {user.username} has been locked.", "warning")
    except Exception:
        db.session.rollback()
        flash("Failed to lock account. Please try again.", "error")
        logger.exception("Error in admin.lock_user: ")

    return redirect(url_for("admin.manage_users"))


@bp.route("/unlock_user/<int:user_id>", methods=["POST"])
@login_required
@roles_required("admin")
def unlock_user(user_id):
    """Admin action to unlock a locked user account."""
    user = User.query.get_or_404(user_id)

    try:
        user.locked_until = None
        user.failed_login_attempts = 0
        db.session.commit()
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} unlocked account for user {user.username} (ID: {user.id})",
                user_id=current_user.id,
                source="admin",
            )
        )
        log_audit_event(
            action="USER_UNLOCKED",
            resource_type="User",
            resource_id=str(user.id),
            details={"username": user.username, "unlocked_by": current_user.username},
        )
        db.session.commit()
        flash(f"Account for {user.username} has been unlocked.", "success")
    except Exception:
        db.session.rollback()
        flash("Failed to unlock account. Please try again.", "error")
        logger.exception("Error in admin.unlock_user: ")

    return redirect(url_for("admin.manage_users"))


@bp.route("/disable_mfa/<int:user_id>", methods=["POST"])
@login_required
@roles_required("admin")
def disable_mfa(user_id):
    """Admin action to disable MFA for a user (e.g. after device loss)."""
    user = User.query.get_or_404(user_id)

    try:
        user.mfa_enabled = False
        user.totp_secret = None
        db.session.commit()
        db.session.add(
            Log(
                level="WARNING",
                message=f"Admin {current_user.username} disabled MFA for user {user.username} (ID: {user.id})",
                user_id=current_user.id,
                source="admin",
            )
        )
        log_audit_event(
            action="MFA_DISABLED_BY_ADMIN",
            resource_type="User",
            resource_id=str(user.id),
            details={"username": user.username, "disabled_by": current_user.username},
        )
        db.session.commit()
        flash(
            f"MFA disabled for {user.username}. They must re-enroll on next login.",
            "warning",
        )
    except Exception:
        db.session.rollback()
        flash("Failed to disable MFA. Please try again.", "error")
        logger.exception("Error in admin.disable_mfa: ")

    return redirect(url_for("admin.manage_users"))


@bp.route("/bulk_user_action", methods=["POST"])
@login_required
@roles_required("admin")
def bulk_user_action():
    """Admin bulk action: lock, unlock, or delete multiple users at once."""
    action = request.form.get("bulk_action")
    user_ids_raw = request.form.getlist("user_ids")

    if not action or not user_ids_raw:
        flash("No action or users selected.", "warning")
        return redirect(url_for("admin.manage_users"))

    try:
        user_ids = [int(uid) for uid in user_ids_raw if uid.isdigit()]
        # Never touch the current admin's own account in bulk
        user_ids = [uid for uid in user_ids if uid != current_user.id]
        users = User.query.filter(User.id.in_(user_ids)).all()

        if not users:
            flash("No valid users selected.", "warning")
            return redirect(url_for("admin.manage_users"))

        affected = 0
        now = datetime.now(timezone.utc)

        for user in users:
            if action == "lock":
                user.locked_until = now + timedelta(days=3650)
                affected += 1
            elif action == "unlock":
                user.locked_until = None
                user.failed_login_attempts = 0
                affected += 1
            elif action == "deactivate":
                user.is_active = False
                affected += 1
            elif action == "activate":
                user.is_active = True
                affected += 1
            elif action == "delete":
                db.session.delete(user)
                affected += 1

        db.session.commit()
        log_audit_event(
            action=f"BULK_{action.upper()}",
            resource_type="User",
            resource_id="bulk",
            details={"user_ids": user_ids, "affected": affected, "action": action},
        )
        db.session.add(
            Log(
                level="WARNING"
                if action in ("lock", "delete", "deactivate")
                else "INFO",
                message=f"Admin {current_user.username} performed bulk {action} on {affected} user(s)",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        flash(f"Bulk {action} applied to {affected} user(s).", "success")
    except Exception:
        db.session.rollback()
        flash("Bulk action failed. Please try again.", "error")
        logger.exception("Error in admin.bulk_user_action: ")

    return redirect(url_for("admin.manage_users"))


@bp.route("/system_overview", methods=["GET"])
@login_required
@roles_required("admin")
def system_overview():
    """Admin page showing system stats."""
    import platform
    import sys

    try:
        from sqlalchemy import text

        user_count = User.query.count()
        role_breakdown = db.session.execute(
            text(
                "SELECT role, COUNT(*) as cnt FROM users GROUP BY role ORDER BY cnt DESC"
            )
        ).fetchall()

        now = datetime.now(timezone.utc)
        locked_count = User.query.filter(User.locked_until > now).count()
        mfa_enabled_count = User.query.filter(User.mfa_enabled == True).count()  # noqa: E712
        inactive_count = User.query.filter(User.is_active == False).count()  # noqa: E712

        recent_logs = Log.query.order_by(Log.timestamp.desc()).limit(5).all()
        error_count = Log.query.filter(Log.level == "ERROR").count()
        warning_count = Log.query.filter(Log.level == "WARNING").count()

        # System resource usage (psutil optional)
        cpu_percent = None
        memory_percent = None
        disk_percent = None
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage("/").percent
        except ImportError:
            pass

        stats = {
            "user_count": user_count,
            "locked_count": locked_count,
            "mfa_enabled_count": mfa_enabled_count,
            "inactive_count": inactive_count,
            "role_breakdown": [{"role": r[0], "count": r[1]} for r in role_breakdown],
            "error_count": error_count,
            "warning_count": warning_count,
            "recent_logs": recent_logs,
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
        }

        logger.info(f"Admin {current_user.id} accessed system overview")
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} (ID: {current_user.id}) accessed system overview",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return render_template(
            "admin/system_overview.html", stats=stats, user_count=user_count
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in admin.system_overview: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading system overview: {e!s}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return redirect(url_for("home"))


@bp.route("/logs", methods=["GET"])
@login_required
@roles_required("admin")
def logs():
    """Admin page showing system logs with pagination and filtering."""

    try:
        page = request.args.get("page", 1, type=int)
        level_filter = request.args.get("level", "").strip()
        source_filter = request.args.get("source", "").strip()
        search_filter = request.args.get("q", "").strip()

        query = Log.query

        if level_filter:
            query = query.filter(Log.level == level_filter.upper())
        if source_filter:
            query = query.filter(Log.source.ilike(f"%{source_filter}%"))
        if search_filter:
            query = query.filter(Log.message.ilike(f"%{search_filter}%"))

        pagination = query.order_by(Log.timestamp.desc()).paginate(
            page=page, per_page=50, error_out=False
        )

        logger.info(f"Admin {current_user.id} viewed system logs")
        db.session.add(
            Log(
                level="INFO",
                message=f"Admin {current_user.username} (ID: {current_user.id}) viewed system logs",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return render_template(
            "admin/logs.html",
            logs=pagination.items,
            pagination=pagination,
            level_filter=level_filter,
            source_filter=source_filter,
            search_filter=search_filter,
        )
    except Exception as e:
        flash("Something went wrong. Please try again.", "error")
        logger.exception("Error in admin.logs: ")
        db.session.add(
            Log(
                level="ERROR",
                message=f"Error loading logs: {e!s}",
                user_id=current_user.id,
                source="admin",
            )
        )
        db.session.commit()
        return redirect(url_for("admin.index"))


@bp.route("/mfa/setup", methods=["GET", "POST"])
@login_required
@roles_required("admin")
def mfa_setup():
    user = current_user
    secret = session.get("mfa_setup_secret")
    if not secret:
        secret = pyotp.random_base32()
        session["mfa_setup_secret"] = secret

    totp = pyotp.TOTP(secret)
    provisioning_uri = totp.provisioning_uri(
        name=user.username, issuer_name="HMIS Hospital"
    )

    img = qrcode.make(provisioning_uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    qr_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if request.method == "POST":
        code = request.form.get("code", "").strip()
        if totp.verify(code):
            user.totp_secret = secret
            user.mfa_enabled = True
            db.session.commit()
            session.pop("mfa_setup_secret", None)
            log_audit_event(
                action="MFA_ENABLED",
                resource_type="User",
                resource_id=str(user.id),
                details={"username": user.username},
            )
            flash("MFA has been successfully enabled for your account!", "success")
            return redirect(url_for("admin.index"))
        else:
            flash("Invalid MFA verification code. Please try again.", "error")

    return render_template("admin/mfa_setup.html", secret=secret, qr_b64=qr_b64)


@bp.route("/admin/audit-trail", methods=["GET"])
@bp.route("/audit-trail", methods=["GET"])
@login_required
@roles_required("admin")
def audit_trail():
    """Admin dashboard view for persistent system audit logs."""
    from departments.models.compliance import AuditLog

    page = request.args.get("page", 1, type=int)
    action_filter = request.args.get("action", "").strip()
    username_filter = request.args.get("username", "").strip()
    resource_type_filter = request.args.get("resource_type", "").strip()

    query = AuditLog.query

    if action_filter:
        query = query.filter(AuditLog.action.ilike(f"%{action_filter}%"))
    if username_filter:
        query = query.filter(AuditLog.username.ilike(f"%{username_filter}%"))
    if resource_type_filter:
        query = query.filter(AuditLog.resource_type.ilike(f"%{resource_type_filter}%"))

    pagination = query.order_by(AuditLog.timestamp.desc()).paginate(
        page=page, per_page=30, error_out=False
    )

    if request.args.get("format") == "json":
        return {
            "total": pagination.total,
            "page": page,
            "pages": pagination.pages,
            "logs": [log.to_dict() for log in pagination.items],
        }

    return render_template(
        "admin/audit_trail.html", pagination=pagination, logs=pagination.items
    )


@bp.route("/audit-trail/export", methods=["GET"])
@login_required
@roles_required("admin")
def export_audit_trail():
    """Export system audit logs as structured JSON for SIEM integration."""
    from departments.models.compliance import AuditLog

    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(1000).all()
    return jsonify(
        {
            "system": "HMIS",
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "count": len(logs),
            "audit_logs": [log_item.to_dict() for log_item in logs],
        }
    )


@bp.route("/admin/outbound-notifications", methods=["GET"])
@bp.route("/outbound-notifications", methods=["GET"])
@login_required
@roles_required("admin")
def outbound_notifications():
    """Admin dashboard view for outbound patient notification logs."""
    from departments.models.notification_log import OutboundNotificationLog

    page = request.args.get("page", 1, type=int)
    event_filter = request.args.get("event_type", "").strip()
    status_filter = request.args.get("status", "").strip()
    recipient_filter = request.args.get("recipient", "").strip()

    query = OutboundNotificationLog.query

    if event_filter:
        query = query.filter(
            OutboundNotificationLog.event_type.ilike(f"%{event_filter}%")
        )
    if status_filter:
        query = query.filter(OutboundNotificationLog.status == status_filter.upper())
    if recipient_filter:
        query = query.filter(
            OutboundNotificationLog.recipient.ilike(f"%{recipient_filter}%")
        )

    pagination = query.order_by(OutboundNotificationLog.created_at.desc()).paginate(
        page=page, per_page=30, error_out=False
    )

    if request.args.get("format") == "json":
        return jsonify(
            {
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
                        "created_at": log.created_at.isoformat()
                        if log.created_at
                        else None,
                    }
                    for log in pagination.items
                ],
            }
        )

    return render_template(
        "admin/outbound_notifications.html",
        logs=pagination.items,
        pagination=pagination,
        event_filter=event_filter,
        status_filter=status_filter,
        recipient_filter=recipient_filter,
    )


@bp.route("/admin/analytics", methods=["GET"])
@bp.route("/analytics", methods=["GET"])
@login_required
@roles_required("admin")
def analytics():
    """Admin dashboard view for hospital-wide executive KPIs and analytics."""
    from departments.admin.analytics import get_executive_kpi_summary

    kpis = get_executive_kpi_summary()

    if request.args.get("format") == "json":
        return jsonify(kpis)

    return render_template("admin/analytics.html", kpis=kpis)


@bp.route("/admin/credentials", methods=["GET", "POST"])
@login_required
@roles_required("admin", "hr")
def staff_credentials():
    """Admin/HR view to manage and list staff credentials sorted by days-until-expiry (soonest first)."""
    from departments.models.hr import StaffCredential

    if request.method == "POST":
        data = request.get_json() or request.form
        staff_name = data.get("staff_name")
        credential_type = data.get("credential_type")
        credential_number = data.get("credential_number")
        expiry_date_str = data.get("expiry_date")

        if (
            not staff_name
            or not credential_type
            or not credential_number
            or not expiry_date_str
        ):
            return jsonify(
                {
                    "error": "staff_name, credential_type, credential_number, and expiry_date required"
                }
            ), 400

        try:
            expiry_date = datetime.strptime(expiry_date_str, "%Y-%m-%d").date()  # noqa: DTZ007
        except ValueError:
            return jsonify({"error": "expiry_date must be formatted YYYY-MM-DD"}), 400

        cred = StaffCredential(
            staff_name=staff_name,
            credential_type=credential_type,
            credential_number=credential_number,
            expiry_date=expiry_date,
            employee_id=data.get("employee_id"),
        )
        db.session.add(cred)
        db.session.commit()
        log_audit_event(
            action="CREDENTIAL_ADDED",
            resource_type="StaffCredential",
            resource_id=str(cred.id),
            details={"staff_name": staff_name, "credential_type": credential_type},
        )
        return jsonify(
            {
                "success": True,
                "credential_id": cred.id,
                "days_until_expiry": cred.days_until_expiry,
            }
        ), 201

    credentials = StaffCredential.query.order_by(
        StaffCredential.expiry_date.asc()
    ).all()

    items = [
        {
            "id": c.id,
            "staff_name": c.staff_name,
            "credential_type": c.credential_type,
            "credential_number": c.credential_number,
            "expiry_date": c.expiry_date.isoformat(),
            "days_until_expiry": c.days_until_expiry,
            "status": "EXPIRED"
            if c.days_until_expiry < 0
            else ("WARNING" if c.days_until_expiry <= 30 else "ACTIVE"),
        }
        for c in credentials
    ]

    if (
        request.args.get("format") == "json"
        or request.headers.get("Accept") == "application/json"
    ):
        return jsonify(items)

    return render_template(
        "admin/credentials.html", credentials=credentials, items=items
    )


@bp.route("/admin/credentials/alerts", methods=["GET"])
@login_required
@roles_required("admin", "hr")
def staff_credential_alerts():
    """API endpoint returning staff credentials expiring within N threshold days (default 30 days)."""
    from departments.models.hr import StaffCredential

    threshold_days = request.args.get("days", 30, type=int)
    cutoff_date = datetime.now(timezone.utc).date() + timedelta(days=threshold_days)

    expiring = (
        StaffCredential.query.filter(StaffCredential.expiry_date <= cutoff_date)
        .order_by(StaffCredential.expiry_date.asc())
        .all()
    )

    return jsonify(
        {
            "threshold_days": threshold_days,
            "count": len(expiring),
            "expiring_credentials": [
                {
                    "id": c.id,
                    "staff_name": c.staff_name,
                    "credential_type": c.credential_type,
                    "credential_number": c.credential_number,
                    "expiry_date": c.expiry_date.isoformat(),
                    "days_until_expiry": c.days_until_expiry,
                }
                for c in expiring
            ],
        }
    ), 200
