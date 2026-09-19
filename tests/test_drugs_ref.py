import pytest
from departments.models.medicine import Medicine
from departments.models.pharmacy import DrugCategory, Drug as PharmDrug
from departments.models.user import User
from extensions import db
from werkzeug.security import generate_password_hash


@pytest.fixture
def medicine_user_id(app):
    """Create test user with medicine role."""
    with app.app_context():
        user = User.query.filter_by(username="doc_drugs_ref_test").first()
        if not user:
            user = User(
                username="doc_drugs_ref_test",
                password=generate_password_hash("password123", method="pbkdf2:sha256"),
                role="medicine",
            )
            db.session.add(user)
            db.session.commit()
        return user.id


@pytest.fixture
def auth_medicine_client(client, medicine_user_id):
    """Authenticated client for medicine department."""
    with client.session_transaction() as sess:
        sess["_user_id"] = str(medicine_user_id)
        sess["_fresh"] = True
    return client


def test_drugs_ref_search_unauthenticated(client):
    """Unauthenticated access to drugs reference search must be rejected."""
    resp = client.get("/medicine/drugs-ref/search")
    assert resp.status_code in [302, 401, 403]


def test_drugs_ref_search_authenticated(auth_medicine_client, app):
    """Authenticated clinician can access drug reference search page."""
    with app.app_context():
        med = Medicine.query.filter_by(generic_name="Paracetamol").first()
        if not med:
            med = Medicine(
                generic_name="Paracetamol",
                brand_name="Panadol",
                dosage="500mg Oral Tablet",
            )
            db.session.add(med)
            db.session.commit()

    resp = auth_medicine_client.get("/medicine/drugs-ref/search?search=Paracetamol")
    assert resp.status_code == 200
    assert b"Paracetamol" in resp.data
    assert b"Panadol" in resp.data


def test_drugs_ref_details_local_fallback(auth_medicine_client, app):
    """Drug details page renders cleanly and uses local fallback when DrugCentral record is absent."""
    with app.app_context():
        med = Medicine.query.filter_by(generic_name="Amoxicillin").first()
        if not med:
            med = Medicine(
                generic_name="Amoxicillin",
                brand_name="Amoxil",
                dosage="250mg/5ml Suspension",
            )
            db.session.add(med)
            db.session.commit()

    resp = auth_medicine_client.get("/medicine/drugs-ref/details/Amoxicillin")
    assert resp.status_code == 200
    assert b"Amoxicillin" in resp.data


def test_fetch_drugs_data_local_fallback(app):
    """fetch_drugs_data falls back to local database records when DrugCentral is unreachable."""
    from departments.medicine.orders import fetch_drugs_data

    with app.app_context():
        # Ensure local medicine exists
        med = Medicine(
            generic_name="IbuprofenTestDrug",
            brand_name="AdvilTestBrand",
            dosage="400mg",
        )
        db.session.add(med)
        db.session.commit()

        results = fetch_drugs_data("IbuprofenTestDrug")
        assert len(results) > 0
        names = [r["generic_name"] for r in results]
        assert "IbuprofenTestDrug" in names


def test_drugs_ref_details_multi_ingredient_string(auth_medicine_client, app):
    """Multi-ingredient compound strings resolve ingredient components instead of failing with 'No details found'."""
    with app.app_context():
        med = Medicine(
            generic_name="Ascorbic acid",
            brand_name="Vitamin C",
            dosage="500mg",
        )
        db.session.add(med)
        db.session.commit()

    complex_query = (
        ".beta.-carotene, ascorbic acid, cholecalciferol, .alpha.-tocopherol acetate, "
        "dl-, thiamine mononitrate, riboflavin, niacinamide, pyridoxine hydrochloride, "
        "folic acid, 5-methyltetrahydrofolic acid, calcium formate, ferrous asparto "
        "glycinate, cyanocobalamin, biotin, potassium iodide, magnesium oxide, zinc oxide and cupric oxide"
    )

    resp = auth_medicine_client.get(f"/medicine/drugs-ref/details/{complex_query}")
    assert resp.status_code == 200
    assert b"No details found" not in resp.data


def test_drugs_ref_autocomplete_endpoint(auth_medicine_client, app):
    """Autocomplete endpoint returns JSON list of matching drug objects."""
    with app.app_context():
        med = Medicine(
            generic_name="LiveSearchDrug",
            brand_name="InstantBrand",
            dosage="100mg",
        )
        db.session.add(med)
        db.session.commit()

    resp = auth_medicine_client.get("/medicine/drugs-ref/api/autocomplete?q=LiveSearch")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["generic_name"] == "LiveSearchDrug"
    assert "url" in data[0]


