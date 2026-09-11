from datetime import datetime

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required

from departments.models.mortuary import MortuaryData
from departments.models.records import Patient
from departments.rbac import roles_required
from extensions import db

from . import bp


@bp.route("/", methods=["GET", "POST"])
@login_required
@roles_required("mortuary", "admin")
def index():
    if request.method == "POST":
        deceased_id = request.form.get("deceased_id", "").strip()
        date_str = request.form.get("date_of_death")
        cause_of_death = request.form.get("cause_of_death", "").strip()

        if not deceased_id or not date_str or not cause_of_death:
            flash("All fields are required for mortuary entry.", "warning")
        else:
            patient = Patient.query.filter_by(patient_id=deceased_id).first()
            if not patient:
                flash(f"Patient ID {deceased_id} not found.", "danger")
            else:
                try:
                    date_of_death = datetime.strptime(date_str, "%Y-%m-%d").date()  # noqa: DTZ007
                    mortuary_rec = MortuaryData(
                        deceased_id=deceased_id,
                        date_of_death=date_of_death,
                        cause_of_death=cause_of_death,
                        recorded_by=current_user.id,
                    )
                    db.session.add(mortuary_rec)
                    db.session.commit()
                    flash(
                        f"Mortuary intake recorded for {deceased_id} successfully.",
                        "success",
                    )
                except Exception as e:  # noqa: BLE001
                    db.session.rollback()
                    flash(f"Error saving record: {e!s}", "danger")

        return redirect(url_for("mortuary.index"))

    records = MortuaryData.query.order_by(MortuaryData.recorded_at.desc()).all()
    return render_template("mortuary.html", records=records)
