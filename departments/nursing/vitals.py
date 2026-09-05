import logging
import sqlite3  # Import sqlite3 module
from datetime import datetime

from flask import flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy.orm import joinedload

from departments.models.admin import Log
from departments.models.nursing import Partogram, Vitals
from departments.models.records import Patient, PatientWaitingList
from departments.rbac import roles_required
from extensions import db

from . import bp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@bp.route('/')
@login_required
@roles_required('nursing', 'medicine', 'admin')
def index():
    """Display the nursing waiting list."""
    try:
        # Fetch all patients in the nursing waiting list who are not yet seen
        nursing_waiting_list = PatientWaitingList.query.filter_by(seen=4).options(
            joinedload(PatientWaitingList.patient)  # Eager load patient details
        ).all()

        # Filter out entries with missing patient relationships
        valid_waiting_list = [
            entry for entry in nursing_waiting_list
            if entry.patient  # Ensure patient relationship exists
        ]

        return render_template('nursing/index.html', waiting_list=valid_waiting_list)
    except Exception as e:
        # Handle any unexpected errors
        flash('Something went wrong. Please try again.', 'error')
        print(f"Error in nursing.index: {e}")  # Debugging
        return redirect(url_for('nursing.index'))
    #vitals
@bp.route('/vitals/<patient_id>', methods=['GET', 'POST'])  # Removed <int:>
@login_required
@roles_required('nursing', 'admin')
def vitals(patient_id):
    """Manage vital signs for a patient."""
    if request.method == 'POST':
        try:
            vitals_data = Vitals(
                patient_id=patient_id,  # String patient_id
                nurse_id=current_user.id,
                temperature=float(request.form['temperature']) if request.form['temperature'] else None,
                pulse=int(request.form['pulse']) if request.form['pulse'] else None,
                blood_pressure_systolic=int(request.form['blood_pressure_systolic']) if request.form['blood_pressure_systolic'] else None,
                blood_pressure_diastolic=int(request.form['blood_pressure_diastolic']) if request.form['blood_pressure_diastolic'] else None,
                respiratory_rate=int(request.form['respiratory_rate']) if request.form['respiratory_rate'] else None,
                oxygen_saturation=int(request.form['oxygen_saturation']) if request.form['oxygen_saturation'] else None,
                blood_glucose=float(request.form['blood_glucose']) if request.form['blood_glucose'] else None,
                weight=float(request.form['weight']) if request.form['weight'] else None,
                height=float(request.form['height']) if request.form['height'] else None
            )
            db.session.add(vitals_data)
            db.session.commit()
            logger.info(f"Nurse {current_user.id} recorded vitals for patient {patient_id}")
            db.session.add(Log(
                level='INFO',
                message=f"Nurse {current_user.username} (ID: {current_user.id}) recorded vitals for patient {patient_id}",
                user_id=current_user.id,
                source='nursing'
            ))
            db.session.commit()
            flash('Vitals recorded successfully.', 'success')
            return redirect(url_for('nursing.vitals', patient_id=patient_id))
        except ValueError as e:
            db.session.rollback()
            flash('Something went wrong. Please try again.', 'error')
            logger.error(f"ValueError in nursing.vitals: {e}", exc_info=True)
            db.session.add(Log(
                level='ERROR',
                message=f"Invalid input error recording vitals: {str(e)}",
                user_id=current_user.id,
                source='nursing'
            ))
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            flash('Something went wrong. Please try again.', 'error')
            logger.error(f"Error in nursing.vitals: {e}", exc_info=True)
            db.session.add(Log(
                level='ERROR',
                message=f"Error recording vitals: {str(e)}",
                user_id=current_user.id,
                source='nursing'
            ))
            db.session.commit()

    try:
        latest_vitals = Vitals.query.filter_by(patient_id=patient_id).order_by(Vitals.timestamp.desc()).first()
        logger.info(f"Nurse {current_user.id} viewed vitals for patient {patient_id}")
        db.session.add(Log(
            level='INFO',
            message=f"Nurse {current_user.username} (ID: {current_user.id}) viewed vitals for patient {patient_id}",
            user_id=current_user.id,
            source='nursing'
        ))
        db.session.commit()
        return render_template('nursing/vitals.html', patient_id=patient_id, vitals=latest_vitals)
    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        logger.error(f"Error in nursing.vitals: {e}", exc_info=True)
        db.session.add(Log(
            level='ERROR',
            message=f"Error loading vitals: {str(e)}",
            user_id=current_user.id,
            source='nursing'
        ))
        db.session.commit()
        return redirect(url_for('home'))


 # Alert/Action line logic
def check_alert_action(dilation, time_hours):
    alert_dilation = 4 + time_hours  # Alert line: starts at 4 cm, increases 1 cm/hour
    action_dilation = 4 + (time_hours - 4) if time_hours >= 4 else 0  # Action line: 4 hours to the right
    status = "Normal"
    if dilation < alert_dilation:
        status = "Crossed Alert Line"
    if time_hours >= 4 and dilation < action_dilation:
        status = "Crossed Action Line"
    return status

# Route to display the Partogram form
@bp.route("/partogram", methods=["GET"])
@login_required
@roles_required('nursing', 'admin')
def record_partogram():
    """Displays the partogram form for recording patient data."""
    try:
        return render_template("nursing/partogram.html")
    except Exception as e:
        flash(f'Error loading partogram form: {str(e)}', 'error')
        print(f"Debug: Error in nursing.record_partogram: {str(e)}")
        return redirect(url_for('nursing.index'))

# Route to submit the Partogram form
@bp.route("/submit_partogram", methods=["POST"])
@login_required
@roles_required('nursing', 'admin')
def submit_partogram():
    """Handles submission of the partogram form and saves to the database."""

    errors = []

    try:
        # Extract form data
        patient_id = request.form.get("patient_id")
        fetal_heart_rate = request.form.get("fetal_heart_rate")
        amniotic_fluid = request.form.get("amniotic_fluid")
        moulding = request.form.get("moulding")
        cervical_dilation = request.form.get("cervical_dilation")
        time_hours = request.form.get("time_hours")
        contractions = request.form.get("contractions")
        oxytocin = request.form.get("oxytocin")
        drugs = request.form.get("drugs")
        pulse = request.form.get("pulse")
        bp = request.form.get("bp")
        temperature = request.form.get("temperature")
        urine_protein = request.form.get("urine_protein")
        urine_volume = request.form.get("urine_volume")
        urine_acetone = request.form.get("urine_acetone")
        timestamp = datetime.utcnow()

        # Step 1: Validate required fields
        required_fields = {
            "Patient ID": patient_id,
            "Fetal Heart Rate": fetal_heart_rate,
            "Amniotic Fluid Status": amniotic_fluid,
            "Moulding": moulding,
            "Cervical Dilation": cervical_dilation,
            "Time (hours)": time_hours,
            "Contractions": contractions,
            "Pulse": pulse,
            "Blood Pressure": bp,
            "Temperature": temperature,
            "Urine Protein": urine_protein,
            "Urine Volume": urine_volume,
            "Urine Acetone": urine_acetone
        }
        for field_name, field_value in required_fields.items():
            if not field_value:
                errors.append(f"{field_name} is required.")

        # Step 2: Validate data types and convert
        if not errors:
            try:
                fetal_heart_rate = int(fetal_heart_rate)
                cervical_dilation = float(cervical_dilation)
                time_hours = float(time_hours)
                contractions = int(contractions)
                oxytocin = float(oxytocin) if oxytocin else 0.0
                pulse = int(pulse)
                if '/' not in bp:
                    errors.append("Blood Pressure must be in the format 'systolic/diastolic' (e.g., 120/80).")
                else:
                    bp_systolic, bp_diastolic = map(int, bp.split('/'))
                temperature = float(temperature)
                urine_volume = int(urine_volume)
            except ValueError as ve:
                errors.append(f"Invalid numeric input: {str(ve)}. Please ensure all numeric fields contain valid numbers.")

        # Step 3: Validate ranges
        if not errors:
            if not (60 <= fetal_heart_rate <= 200):
                errors.append("Fetal Heart Rate must be between 60 and 200 beats per minute.")
            if not (0 <= cervical_dilation <= 10):
                errors.append("Cervical Dilation must be between 0 and 10 cm.")
            if not (0 <= time_hours):
                errors.append("Time since active labor start cannot be negative.")
            if not (0 <= contractions <= 5):
                errors.append("Contractions per 10 minutes must be between 0 and 5.")
            if oxytocin and oxytocin < 0:
                errors.append("Oxytocin units cannot be negative.")
            if not (40 <= pulse <= 200):
                errors.append("Pulse must be between 40 and 200 beats per minute.")
            if not (50 <= bp_systolic <= 200):
                errors.append("Systolic Blood Pressure must be between 50 and 200 mmHg.")
            if not (30 <= bp_diastolic <= 150):
                errors.append("Diastolic Blood Pressure must be between 30 and 150 mmHg.")
            if bp_systolic <= bp_diastolic:
                errors.append("Systolic Blood Pressure must be greater than Diastolic Blood Pressure.")
            if not (35 <= temperature <= 42):
                errors.append("Temperature must be between 35 and 42 °C.")
            if not (0 <= urine_volume):
                errors.append("Urine Volume cannot be negative.")

        # Step 4: Validate categorical fields
        if not errors:
            valid_amniotic_fluid = ['clear', 'meconium', 'blood', 'absent']
            if amniotic_fluid not in valid_amniotic_fluid:
                errors.append(f"Amniotic Fluid Status must be one of: {', '.join(valid_amniotic_fluid)}.")

            valid_moulding = ['none', '1+', '2+', '3+']
            if moulding not in valid_moulding:
                errors.append(f"Moulding must be one of: {', '.join(valid_moulding)}.")

            valid_urine_levels = ['negative', '1+', '2+', '3+']
            if urine_protein not in valid_urine_levels:
                errors.append(f"Urine Protein must be one of: {', '.join(valid_urine_levels)}.")
            if urine_acetone not in valid_urine_levels:
                errors.append(f"Urine Acetone must be one of: {', '.join(valid_urine_levels)}.")

        # Step 5: Validate string lengths (optional fields)
        if not errors and drugs and len(drugs) > 100:
            errors.append("Drugs field cannot exceed 100 characters.")

        # Step 6: Check for increasing time_hours for the same patient
        if not errors:
            previous_entry = Partogram.query.filter_by(patient_id=patient_id).order_by(Partogram.time_hours.desc()).first()
            if previous_entry and time_hours <= previous_entry.time_hours:
                errors.append("Time since active labor start must be greater than the previous entry for this patient.")

        # If there are errors, render the error page
        if errors:
            return render_template('nursing/error.html', errors=errors)

        # Step 7: Check alert/action lines
        labour_status = check_alert_action(cervical_dilation, time_hours)

        # Step 8: Create and save the Partogram entry
        new_partogram = Partogram(
            patient_id=patient_id,
            fetal_heart_rate=fetal_heart_rate,
            amniotic_fluid=amniotic_fluid,
            moulding=moulding,
            cervical_dilation=cervical_dilation,
            time_hours=time_hours,
            contractions=contractions,
            oxytocin=oxytocin,
            drugs=drugs,
            pulse=pulse,
            bp_systolic=bp_systolic,
            bp_diastolic=bp_diastolic,
            temperature=temperature,
            urine_protein=urine_protein,
            urine_volume=urine_volume,
            urine_acetone=urine_acetone,
            labour_status=labour_status,
            timestamp=timestamp,
            recorded_by=current_user.id
        )

        db.session.add(new_partogram)
        db.session.commit()

        flash("Partogram recorded successfully!", "success")
        return redirect(url_for("nursing.view_partogram", patient_id=patient_id))

    except ValueError as ve:
        db.session.rollback()
        return render_template('nursing/error.html', errors=[f"Invalid input: {str(ve)}"])
    except Exception as e:
        db.session.rollback()
        return render_template('nursing/error.html', errors=[f"Error saving partogram: {str(e)}"])

@bp.route('/view_partogram/<patient_id>', methods=['GET'])
@login_required
@roles_required('nursing', 'admin')
def view_partogram(patient_id):
    """Displays partogram records for a specific patient, including graphs."""
    conn = db.engine.raw_connection()
    cursor = conn.cursor()

    try:
        # Fetch partogram records for the specified patient
        cursor.execute('''
            SELECT
                time_hours,
                cervical_dilation,
                fetal_heart_rate,
                contractions,
                pulse,
                bp_systolic,
                bp_diastolic,
                temperature,
                amniotic_fluid,
                moulding,
                labour_status,
                urine_protein,
                urine_volume,
                urine_acetone,
                timestamp,
                recorded_by
            FROM partogram
            WHERE patient_id = ?
            ORDER BY time_hours
        ''', (patient_id,))
        rows = cursor.fetchall()

        if not rows:
            flash(f'No partogram records found for patient {patient_id}.', 'info')
            return redirect(url_for('nursing.index'))

        # Convert rows to a list of dictionaries for easier template rendering
        entries = [
            {
                'time_hours': row[0],
                'cervical_dilation': row[1],
                'fetal_heart_rate': row[2],
                'contractions': row[3],
                'pulse': row[4],
                'bp_systolic': row[5],
                'bp_diastolic': row[6],
                'temperature': row[7],
                'amniotic_fluid': row[8],
                'moulding': row[9],
                'labour_status': row[10],
                'urine_protein': row[11],
                'urine_volume': row[12],
                'urine_acetone': row[13],
                'timestamp': row[14],  # ISO format string
                'recorded_by': row[15]
            }
            for row in rows
        ]

        # Fetch usernames for recorded_by mapping
        recorded_by_ids = [entry['recorded_by'] for entry in entries if entry['recorded_by']]
        if recorded_by_ids:
            cursor.execute('SELECT id, username FROM users WHERE id IN ({})'.format(','.join('?' * len(recorded_by_ids))), recorded_by_ids)  # nosec B608
            user_rows = cursor.fetchall()
            user_name_map = {row[0]: row[1] for row in user_rows}
        else:
            user_name_map = {}

        # Debugging
        print(f"Debug: Found {len(entries)} partogram records for patient {patient_id}")

        return render_template(
            'nursing/view_partogram.html',
            partogram_data=entries,
            patient_id=patient_id,
            user_name_map=user_name_map
        )

    except sqlite3.Error as e:
        flash(f'Error fetching partogram records: {str(e)}', 'error')
        print(f"Debug: Error in nursing.view_partogram: {str(e)}")
        return redirect(url_for('nursing.index'))
    finally:
        cursor.close()
        conn.close()
# Route to view all partograms (paginated)
@bp.route("/view_partograms")
@login_required
@roles_required('nursing', 'admin')
def view_partograms():

    # Get query parameters
    patient_id = request.args.get('patient_id', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 10  # Number of records per page

    from extensions import db
    conn = db.engine.raw_connection()
    cursor = conn.cursor()

    try:
        # Fetch usernames for recorded_by mapping
        cursor.execute('SELECT id, username FROM users')
        user_name_map = {row[0]: row[1] for row in cursor.fetchall()}

        # Build the query to count total unique patients (for pagination)
        count_query = 'SELECT COUNT(DISTINCT patient_id) FROM partogram'
        count_params = []
        if patient_id:
            count_query += ' WHERE patient_id = ?'
            count_params.append(patient_id)

        cursor.execute(count_query, count_params)
        total_records = cursor.fetchone()[0]

        # Calculate pagination details
        total_pages = (total_records + per_page - 1) // per_page
        page = max(1, min(page, total_pages))  # Ensure page is within valid range
        offset = (page - 1) * per_page

        # Build the query to fetch the most recent partogram record for each patient
        select_query = '''
            SELECT id, patient_id, timestamp, recorded_by
            FROM partogram p1
            WHERE timestamp = (
                SELECT MAX(timestamp)
                FROM partogram p2
                WHERE p2.patient_id = p1.patient_id
            )
        '''
        select_params = []
        if patient_id:
            select_query += ' AND patient_id = ?'
            select_params.append(patient_id)
        select_query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
        select_params.extend([per_page, offset])

        cursor.execute(select_query, select_params)
        rows = cursor.fetchall()

        # Convert rows to a list of dictionaries
        partograms_list = [
            {
                'id': row[0],
                'patient_id': row[1],
                'timestamp': row[2],  # We'll parse this in the template
                'recorded_by': row[3]
            }
            for row in rows
        ]

        # Create a pagination object (mimicking Flask-SQLAlchemy's pagination)
        class Pagination:
            def __init__(self, page, per_page, total, items):
                self.page = page
                self.per_page = per_page
                self.total = total
                self.items = items

            @property
            def has_prev(self):
                return self.page > 1

            @property
            def has_next(self):
                return self.page < self.total_pages

            @property
            def prev_num(self):
                return self.page - 1

            @property
            def next_num(self):
                return self.page + 1

            @property
            def total_pages(self):
                return (self.total + self.per_page - 1) // self.per_page

            def iter_pages(self, left_edge=2, left_current=2, right_current=5, right_edge=2):
                last = 0
                for num in range(1, self.total_pages + 1):
                    if (num <= left_edge or
                        (num > self.page - left_current - 1 and num < self.page + right_current) or
                        num > self.total_pages - right_edge):
                        if last + 1 != num:
                            yield None
                        yield num
                        last = num

        partograms = Pagination(page, per_page, total_records, partograms_list)

        return render_template(
            'nursing/view_partograms.html',
            partograms=partograms,
            user_name_map=user_name_map
        )

    except sqlite3.Error as e:
        print(f"Debug: Error in nursing.view_partograms: {str(e)}")
        return render_template('nursing/error.html', errors=[f"Error fetching partogram records: {str(e)}"])
    finally:
        cursor.close()
        conn.close()


@bp.route('/vital_signs', methods=['GET', 'POST'])
@login_required
@roles_required('nursing', 'admin', 'medicine')
def vital_signs():

    if request.method == 'POST':
        try:
            patient_id = request.form.get('patient_id')
            if not patient_id:
                flash('Patient ID is required.', 'error')
                return redirect(url_for('nursing.vital_signs'))

            # Capture all fields from the Vitals model
            heart_rate = request.form.get('heart_rate')
            respiratory_rate = request.form.get('respiratory_rate')
            oxygen_saturation = request.form.get('oxygen_saturation')
            temperature = request.form.get('temperature')
            pulse = request.form.get('pulse')
            blood_pressure_systolic = request.form.get('blood_pressure_systolic')
            blood_pressure_diastolic = request.form.get('blood_pressure_diastolic')
            blood_glucose = request.form.get('blood_glucose')
            weight = request.form.get('weight')
            height = request.form.get('height')

            new_vital_sign = Vitals(
                patient_id=patient_id,
                heart_rate=int(heart_rate) if heart_rate else None,
                respiratory_rate=int(respiratory_rate) if respiratory_rate else None,
                oxygen_saturation=int(oxygen_saturation) if oxygen_saturation else None,
                temperature=float(temperature) if temperature else None,
                pulse=int(pulse) if pulse else None,
                blood_pressure_systolic=int(blood_pressure_systolic) if blood_pressure_systolic else None,
                blood_pressure_diastolic=int(blood_pressure_diastolic) if blood_pressure_diastolic else None,
                blood_glucose=float(blood_glucose) if blood_glucose else None,
                weight=float(weight) if weight else None,
                height=float(height) if height else None,
                timestamp=datetime.utcnow(),
                recorded_by=current_user.id
            )
            db.session.add(new_vital_sign)
            db.session.commit()
            flash('Vital signs recorded.', 'success')
            return redirect(url_for('nursing.vital_signs'))
        except ValueError:
            flash('Invalid input: Please ensure numeric fields contain valid numbers.', 'error')
            return redirect(url_for('nursing.vital_signs'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error recording vital signs: {str(e)}', 'error')
            return redirect(url_for('nursing.vital_signs'))

    return render_template('nursing/vital_signs.html')


@bp.route('/vitals_chart/<patient_id>')
@login_required
@roles_required('nursing', 'admin', 'medicine')
def vitals_chart(patient_id):

    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    vitals_history = Vitals.query.filter_by(patient_id=patient_id).order_by(Vitals.timestamp.asc()).all()

    timestamps = [v.timestamp.strftime('%m-%d %H:%M') for v in vitals_history]
    temps = [v.temperature or 0 for v in vitals_history]
    pulse = [v.pulse or 0 for v in vitals_history]
    bp_sys = [v.blood_pressure_systolic or 0 for v in vitals_history]
    bp_dia = [v.blood_pressure_diastolic or 0 for v in vitals_history]
    spo2 = [v.oxygen_saturation or 0 for v in vitals_history]

    return render_template(
        'nursing/vitals_chart.html',
        patient=patient,
        vitals_history=vitals_history,
        timestamps=timestamps,
        temps=temps,
        pulse=pulse,
        bp_sys=bp_sys,
        bp_dia=bp_dia,
        spo2=spo2
    )


# ─────────────────────────────────────────────
# FLUID BALANCE CHART
# ─────────────────────────────────────────────

@bp.route('/fluid_balance/<patient_id>', methods=['GET', 'POST'])
@login_required
@roles_required('nursing', 'admin', 'medicine')
def fluid_balance(patient_id):

    patient = Patient.query.filter_by(patient_id=patient_id).first_or_404()
    return render_template('nursing/fluid_balance.html', patient=patient)
