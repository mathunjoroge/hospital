from flask import render_template, redirect, url_for, request, flash
from flask_login import login_required, current_user
from extensions import db
from departments.models.mortuary import MortuaryData
from departments.models.records import Patient
from datetime import datetime
from . import bp

@bp.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if current_user.role not in ['mortuary', 'admin']:
        flash("Unauthorized access to Mortuary Department.", "danger")
        return redirect(url_for('home'))

    if request.method == 'POST':
        deceased_id = request.form.get('deceased_id', '').strip()
        date_str = request.form.get('date_of_death')
        cause_of_death = request.form.get('cause_of_death', '').strip()

        if not deceased_id or not date_str or not cause_of_death:
            flash("All fields are required for mortuary entry.", "warning")
        else:
            patient = Patient.query.filter_by(patient_id=deceased_id).first()
            if not patient:
                flash(f"Patient ID {deceased_id} not found.", "danger")
            else:
                try:
                    date_of_death = datetime.strptime(date_str, '%Y-%m-%d').date()
                    mortuary_rec = MortuaryData(
                        deceased_id=deceased_id,
                        date_of_death=date_of_death,
                        cause_of_death=cause_of_death,
                        recorded_by=current_user.id
                    )
                    db.session.add(mortuary_rec)
                    db.session.commit()
                    flash(f"Mortuary intake recorded for {deceased_id} successfully.", "success")
                except Exception as e:
                    db.session.rollback()
                    flash(f"Error saving record: {str(e)}", "danger")

        return redirect(url_for('mortuary.index'))

    records = MortuaryData.query.order_by(MortuaryData.recorded_at.desc()).all()
    return render_template('mortuary.html', records=records)