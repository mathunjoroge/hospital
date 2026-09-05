from datetime import datetime, timedelta

import pytest
from werkzeug.security import generate_password_hash

from departments.admin.analytics import (
    get_bed_occupancy_stats,
    get_executive_kpi_summary,
    get_inpatient_admission_trends,
    get_insurance_claims_stats,
    get_revenue_summary,
)
from departments.models.billing import Invoice, PaidBill
from departments.models.insurance import Claim, InsuranceScheme, PatientInsurance
from departments.models.medicine import AdmittedPatient, Ward
from departments.models.records import Patient
from departments.models.user import User
from extensions import db


@pytest.fixture
def sample_data(app):
    """Seed test data for wards, beds, admissions, billings, and claims."""
    with app.app_context():
        # Patient
        p = Patient(
            patient_id="P-ANAL-001",
            name="Alice Smith",
            sex="Female",
            date_of_birth=datetime(1995, 5, 12).date(),
            marital_status="Single",
            contact="0711223344",
            place_of_residence="Nairobi",
            national_id="NAT-888999",
            next_of_kin="Bob Smith",
            relationship_with_next_of_kin="Brother",
            next_of_kin_contact="0711223355",
            emergency_contact="0711223355",
        )
        db.session.add(p)
        db.session.commit()

        # Wards
        ward1 = Ward(name="General Female Ward", sex="Female", number_of_beds=5, daily_charge=1500.0)
        ward2 = Ward(name="VIP Private Ward", sex="Mixed", number_of_beds=2, daily_charge=5000.0)
        db.session.add_all([ward1, ward2])
        db.session.commit()

        # Admitted Patients (1 active in ward 1, 1 discharged in ward 1, 1 active in ward 2)
        adm1 = AdmittedPatient(
            patient_id=p.patient_id,
            ward_id=ward1.id,
            admitted_on=datetime.utcnow() - timedelta(days=2),
            admission_criteria="Acute Severe Fever",
            admitted_by=1,
            discharged_on=None,
        )
        adm2 = AdmittedPatient(
            patient_id=p.patient_id,
            ward_id=ward1.id,
            admitted_on=datetime.utcnow() - timedelta(days=10),
            admission_criteria="Pneumonia",
            admitted_by=1,
            discharged_on=datetime.utcnow() - timedelta(days=5),
        )
        adm3 = AdmittedPatient(
            patient_id=p.patient_id,
            ward_id=ward2.id,
            admitted_on=datetime.utcnow() - timedelta(days=1),
            admission_criteria="Post-Op Observation",
            admitted_by=1,
            discharged_on=None,
        )
        db.session.add_all([adm1, adm2, adm3])
        db.session.commit()

        # Revenue
        pb1 = PaidBill(
            receipt_number="REC-A1",
            patient_id=p.patient_id,
            grand_total=3000.0,
            amount_paid=3000.0,
            balance=0.0,
            payment_method="MPESA",
        )
        pb2 = PaidBill(
            receipt_number="REC-A2",
            patient_id=p.patient_id,
            grand_total=1500.0,
            amount_paid=1500.0,
            balance=0.0,
            payment_method="CASH",
        )
        db.session.add_all([pb1, pb2])
        db.session.commit()

        # Claims
        scheme = InsuranceScheme(name="SHA Health", code="SHA-ANAL", scheme_type="public")
        db.session.add(scheme)
        db.session.commit()

        pins = PatientInsurance(
            patient_id=p.patient_id, scheme_id=scheme.id, member_number="MEM-100"
        )
        db.session.add(pins)
        db.session.commit()

        inv = Invoice(patient_id=p.patient_id, total_amount=4500.0, status="UNPAID")
        db.session.add(inv)
        db.session.commit()

        cl1 = Claim(
            claim_number="CLM-ANAL-01",
            invoice_id=inv.id,
            patient_id=p.patient_id,
            scheme_id=scheme.id,
            claimed_amount=4500.0,
            approved_amount=4000.0,
            status="APPROVED",
        )
        db.session.add(cl1)
        db.session.commit()

        return p.patient_id


def test_bed_occupancy_stats_math(app, sample_data):
    """Verify bed occupancy calculation logic."""
    with app.app_context():
        stats = get_bed_occupancy_stats()
        assert stats['total_beds'] == 7
        assert stats['occupied_beds'] == 2
        assert stats['available_beds'] == 5
        assert stats['occupancy_rate'] == 28.6
        assert len(stats['ward_breakdown']) == 2


def test_inpatient_admission_trends(app, sample_data):
    """Verify 30-day admission trend series generation."""
    with app.app_context():
        trends = get_inpatient_admission_trends(days=30)
        assert len(trends) == 30
        total_adm_count = sum(t['admissions'] for t in trends)
        assert total_adm_count == 3


def test_revenue_and_claims_stats(app, sample_data):
    """Verify revenue aggregation and insurance claims stats."""
    with app.app_context():
        rev = get_revenue_summary()
        assert rev['total_collected'] == 4500.0
        assert rev['by_method'].get('MPESA') == 3000.0
        assert rev['by_method'].get('CASH') == 1500.0

        claims = get_insurance_claims_stats()
        assert claims['total_claims'] == 1
        assert claims['approved_claims'] == 1
        assert claims['approval_rate'] == 100.0


def test_executive_kpi_summary(app, sample_data):
    """Verify consolidated KPI payload structure."""
    with app.app_context():
        kpis = get_executive_kpi_summary()
        assert 'bed_occupancy' in kpis
        assert 'admission_trends' in kpis
        assert 'revenue_summary' in kpis
        assert 'claims_stats' in kpis


def test_admin_analytics_route_rbac(client, app):
    """Verify /admin/analytics endpoint requires admin login."""
    with app.app_context():
        staff_user = User(
            username="nurse_analytics_test",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="nursing",
        )
        admin_user = User(
            username="admin_analytics_test",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="admin",
        )
        db.session.add_all([staff_user, admin_user])
        db.session.commit()

    # Login as nursing (non-admin)
    client.post("/login", data={"username": "nurse_analytics_test", "password": "Password123!"})
    resp_staff = client.get("/admin/analytics")
    assert resp_staff.status_code in (403, 302)

    # Login as admin
    client.post("/login", data={"username": "admin_analytics_test", "password": "Password123!"})
    resp_admin = client.get("/admin/analytics")
    assert resp_admin.status_code == 200
    assert b"Executive Analytics Dashboard" in resp_admin.data


def test_admin_analytics_json_export(client, app, sample_data):
    """Verify /admin/analytics?format=json returns valid JSON payload."""
    with app.app_context():
        admin = User(
            username="admin_analytics_json",
            password=generate_password_hash("Password123!", method="pbkdf2:sha256"),
            role="admin",
        )
        db.session.add(admin)
        db.session.commit()

    client.post("/login", data={"username": "admin_analytics_json", "password": "Password123!"})
    resp = client.get("/admin/analytics?format=json")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "bed_occupancy" in data
    assert "admission_trends" in data
    assert data["bed_occupancy"]["occupied_beds"] == 2
