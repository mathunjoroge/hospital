import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

from flask import (
    flash,
    g,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from departments.api.audit import log_audit_event
from departments.models.patient_user import PatientUser
from departments.models.records import Patient
from departments.notifications.triggers import trigger_password_reset_email
from extensions import db

from . import patient_portal_bp


def _as_utc(dt: "datetime | None") -> "datetime | None":
    """
    Return dt as a UTC-aware datetime.

    SQLite strips timezone info on roundtrip even with DateTime(timezone=True).
    Any naive datetime stored by this app was written as UTC, so we re-attach
    UTC here before comparing against datetime.now(timezone.utc).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def patient_login_required(f):
    """Decorator ensuring request is made by an authenticated PatientUser session."""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        patient_user_id = session.get("patient_user_id")
        if not patient_user_id:
            flash("Please log in to access the patient portal.", "warning")
            return redirect(url_for("patient_portal.login"))

        patient_user = PatientUser.query.get(patient_user_id)
        if not patient_user or not patient_user.is_active:
            session.pop("patient_user_id", None)
            flash("Your portal session is invalid or inactive.", "danger")
            return redirect(url_for("patient_portal.login"))

        if patient_user.is_locked():
            session.pop("patient_user_id", None)
            flash("Account is locked due to multiple failed login attempts.", "danger")
            return redirect(url_for("patient_portal.login"))

        g.current_patient_user = patient_user
        g.current_patient = patient_user.patient
        return f(*args, **kwargs)

    return decorated_function


@patient_portal_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("patient_user_id"):
        return redirect(url_for("patient_portal.dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("patient_portal/login.html")

        user = PatientUser.query.filter_by(username=username).first()
        if not user:
            flash("Invalid credentials.", "danger")
            return render_template("patient_portal/login.html")

        if user.is_locked():
            flash(
                "Account is locked due to consecutive failed attempts. Try again later.",
                "danger",
            )
            return render_template("patient_portal/login.html")

        if not user.check_password(password):
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=15)
                flash(
                    "Account locked for 15 minutes due to 5 failed login attempts.",
                    "danger",
                )
            else:
                flash("Invalid credentials.", "danger")
            db.session.commit()
            return render_template("patient_portal/login.html")

        # Successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        db.session.commit()

        session["patient_user_id"] = user.id
        flash(f"Welcome back, {user.patient.name}!", "success")
        return redirect(url_for("patient_portal.dashboard"))

    return render_template("patient_portal/login.html")


@patient_portal_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("patient_user_id"):
        return redirect(url_for("patient_portal.dashboard"))

    if request.method == "POST":
        national_id = (request.form.get("national_id") or "").strip()
        patient_id = (request.form.get("patient_id") or "").strip()
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not password:
            flash("All required fields must be filled out.", "danger")
            return render_template("patient_portal/register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("patient_portal/register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return render_template("patient_portal/register.html")

        # Look up existing clinical Patient
        patient = None
        if national_id:
            patients = Patient.query.filter(Patient.national_id.isnot(None)).all()
            patient = next((p for p in patients if p.national_id == national_id), None)
        elif patient_id and patient_id.isdigit():
            patient = Patient.query.get(int(patient_id))

        if not patient:
            flash(
                "No matching patient record found in hospital registry. Please verify National ID or Patient ID.",
                "danger",
            )
            return render_template("patient_portal/register.html")

        if patient.portal_user:
            flash(
                "A portal account already exists for this patient. Please log in.",
                "warning",
            )
            return redirect(url_for("patient_portal.login"))

        if PatientUser.query.filter_by(username=username).first():
            flash("Username is already taken. Please choose another.", "danger")
            return render_template("patient_portal/register.html")

        # Create PatientUser
        new_user = PatientUser(patient_id=patient.id, username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        flash(
            "Registration successful! Please log in with your credentials.", "success"
        )
        return redirect(url_for("patient_portal.login"))

    return render_template("patient_portal/register.html")


@patient_portal_bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("patient_user_id", None)
    flash("You have been logged out of the patient portal.", "info")
    return redirect(url_for("patient_portal.login"))


@patient_portal_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    """Initiate password reset: generate a secure token and dispatch reset email."""
    if session.get("patient_user_id"):
        return redirect(url_for("patient_portal.dashboard"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()

        if username:
            user = PatientUser.query.filter_by(username=username).first()
            if user and user.is_active:
                token = secrets.token_urlsafe(32)
                user.reset_token = token
                user.reset_token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)
                db.session.commit()

                reset_link = url_for(
                    "patient_portal.reset_password", token=token, _external=True
                )
                trigger_password_reset_email(user, reset_link)

        # Always show success to prevent email/username enumeration
        flash(
            "If an account with that username exists, a password reset link has been sent.",
            "success",
        )
        return redirect(url_for("patient_portal.login"))

    return render_template("patient_portal/forgot_password.html")


@patient_portal_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    """Validate reset token and allow patient to set a new password."""
    if session.get("patient_user_id"):
        return redirect(url_for("patient_portal.dashboard"))

    user = PatientUser.query.filter_by(reset_token=token).first()

    # Validate token existence and expiry
    if (
        not user
        or not user.reset_token_expiry
        or datetime.now(timezone.utc) > _as_utc(user.reset_token_expiry)
    ):
        flash("This password reset link is invalid or has expired.", "danger")
        return redirect(url_for("patient_portal.login"))

    if request.method == "POST":
        new_password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not new_password or new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("patient_portal/reset_password.html", token=token)

        if len(new_password) < 8:
            flash("Password must be at least 8 characters long.", "danger")
            return render_template("patient_portal/reset_password.html", token=token)

        # Update password, clear token, and reset any lockout state
        user.set_password(new_password)
        user.reset_token = None
        user.reset_token_expiry = None
        user.failed_login_attempts = 0
        user.locked_until = None
        db.session.commit()

        log_audit_event(
            action="PASSWORD_RESET",
            resource_type="PatientUser",
            resource_id=str(user.id),
            details=f"Password successfully reset via token for patient_id={user.patient_id}",
            user_id=user.id,
        )

        flash("Your password has been successfully reset. Please log in.", "success")
        return redirect(url_for("patient_portal.login"))

    return render_template("patient_portal/reset_password.html", token=token)
