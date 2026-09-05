import csv
import json
import logging
from collections import Counter
from datetime import datetime, timedelta
from io import StringIO

from flask import (
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import login_required

from departments.models.billing import DrugsBill
from departments.models.pharmacy import (  # Import PatientWaitingList and Patient models
    Batch,
    DispensedDrug,
    Drug,
    Expiry,
)
from departments.rbac import roles_required
from extensions import db

from . import bp  # Import the blueprint

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@bp.route('/expiries_report', methods=['GET', 'POST'])
@login_required
@roles_required('pharmacy', 'admin')
def expiries_report():
    """Generates a report of expired batches removed within a date range."""
    try:
        # Default date range: last 30 days
        default_end = datetime.today().date()
        default_start = default_end - timedelta(days=30)

        start_date = request.form.get('start_date', default_start.strftime('%Y-%m-%d'))
        end_date = request.form.get('end_date', default_end.strftime('%Y-%m-%d'))

        # Convert string dates to datetime objects
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format. Use YYYY-MM-DD.', 'error')
            start_date, end_date = default_start, default_end

        # Query expiries within the date range, joining with drugs for name
        report_data = db.session.query(
            Drug.generic_name,
            Drug.brand_name,
            Expiry.batch_number,
            Expiry.quantity_removed,
            Expiry.expiry_date,
            Expiry.removal_date
        ).join(
            Drug, Expiry.drug_id == Drug.id
        ).filter(
            Expiry.removal_date >= start_date,
            Expiry.removal_date <= end_date
        ).order_by(
            Expiry.removal_date.desc()
        ).all()

        # Calculate total quantity removed
        total_removed = sum(item.quantity_removed for item in report_data)

        return render_template(
            'pharmacy/expiries_report.html',
            report_data=report_data,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            total_removed=total_removed
        )

    except Exception as e:
        flash('Something went wrong. Please try again.', 'error')
        print(f"Debug: Error in pharmacy.expiries_report: {e}")
        return redirect(url_for('pharmacy.index'))


@bp.route('/analytics', methods=['GET', 'POST'])
@login_required
def analytics():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    if request.method == 'POST':
        start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')

    # Sales Trends (from drugs_bill, only paid bills)
    sales = DrugsBill.query.filter(
        DrugsBill.billed_at.between(start_date, end_date),
        DrugsBill.status == 1
    ).all()
    sales_data = {}
    for sale in sales:
        date_key = sale.billed_at.strftime('%Y-%m-%d')
        sales_data[date_key] = sales_data.get(date_key, 0) + float(sale.total_cost)

    # Top Dispensed Drugs (with additional fields)
    dispensed = DispensedDrug.query.filter(
        DispensedDrug.date_dispensed.between(start_date, end_date)
    ).join(Drug, DispensedDrug.drug_id == Drug.id).all()
    drug_counts = Counter([d.drug.generic_name for d in dispensed]).most_common(5)
    top_drugs = []
    for name, count in drug_counts:
        drug = Drug.query.filter_by(generic_name=name).first()  # Get drug details
        top_drugs.append({
            'generic_name': name,
            'brand_name': drug.brand_name if drug else 'N/A',
            'strength': drug.strength if drug else 'N/A',
            'dosage_form': drug.dosage_form if drug else 'N/A',
            'count': count
        })

    # Inventory Usage Rate (with additional fields)
    drugs = Drug.query.all()
    usage_rates = []
    for drug in drugs:
        dispensed_qty = db.session.query(
            db.func.sum(DispensedDrug.quantity_dispensed)
        ).filter(
            DispensedDrug.drug_id == drug.id,
            DispensedDrug.date_dispensed.between(start_date, end_date)
        ).scalar() or 0
        initial_stock = drug.quantity_in_stock + dispensed_qty
        usage = (dispensed_qty / initial_stock * 100) if initial_stock > 0 else 0
        usage_rates.append({
            'generic_name': drug.generic_name,
            'brand_name': drug.brand_name or 'N/A',
            'strength': drug.strength,
            'dosage_form': drug.dosage_form,
            'remaining': drug.quantity_in_stock,
            'usage': usage
        })

    # Expiry Risks (with additional fields)
    expiry_risks = Batch.query.filter(
        Batch.expiry_date <= (datetime.now() + timedelta(days=90)),
        Batch.quantity_in_stock > 0
    ).join(Drug, Batch.drug_id == Drug.id).all()
    expiry_data = [
        {
            'generic_name': batch.drug.generic_name,
            'brand_name': batch.drug.brand_name or 'N/A',
            'strength': batch.drug.strength,
            'dosage_form': batch.drug.dosage_form,
            'batch': batch.batch_number,
            'expiry': batch.expiry_date.strftime('%Y-%m-%d'),
            'qty': batch.quantity_in_stock
        }
        for batch in expiry_risks
    ]

    # Peak Dispensing Hours
    peak_hours = Counter([d.date_dispensed.hour for d in dispensed])
    hours_data = [{'hour': h, 'count': peak_hours.get(h, 0)} for h in range(24)]

    # Store sales_data in session for export
    session['sales_data'] = sales_data
    session['start_date'] = start_date.strftime('%Y-%m-%d')
    session['end_date'] = end_date.strftime('%Y-%m-%d')

    return render_template(
        'pharmacy/analytics.html',
        sales_data=json.dumps(list(sales_data.items())),
        top_drugs=top_drugs,
        usage_rates=usage_rates,
        expiry_data=expiry_data,
        hours_data=json.dumps(hours_data),
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

@bp.route('/analytics/export')
@login_required
def export_analytics():
    sales_data = session.get('sales_data', {})
    start_date = session.get('start_date', (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'))
    end_date = session.get('end_date', datetime.now().strftime('%Y-%m-%d'))

    if not sales_data:
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        sales = DrugsBill.query.filter(
            DrugsBill.billed_at.between(start_date_dt, end_date_dt),
            DrugsBill.status == 1
        ).all()
        sales_data = {}
        for sale in sales:
            date_key = sale.billed_at.strftime('%Y-%m-%d')
            sales_data[date_key] = sales_data.get(date_key, 0) + float(sale.total_cost)

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Sales'])
    for date, total in sales_data.items():
        writer.writerow([date, total])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={"Content-Disposition": f"attachment;filename=analytics_sales_{start_date}_to_{end_date}.csv"}
    )

# --- AI Drug Discovery Routes ---

