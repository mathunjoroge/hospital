"""T3.x — UI Wiring: Active Encounters Summary tests."""
import pytest

from app import app as flask_app
from departments.models.encounter import Encounter
from departments.models.user import User
from extensions import db


@pytest.fixture
def authenticated_client():
    """Fixture that provides a test client with an authenticated admin session."""
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    flask_app.config["RATELIMIT_ENABLED"] = False
    flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"

    with flask_app.app_context():
        db.engine.dispose()
        db.create_all()

        # Create a test user
        user = User(id=99, username="test_admin", password="x", role="admin")
        db.session.add(user)
        db.session.commit()

        with flask_app.test_client() as test_client:
            with test_client.session_transaction() as sess:
                sess["_user_id"] = "99"
                sess["_fresh"] = True
            yield test_client

        db.session.remove()
        db.drop_all()
        db.engine.dispose()

def test_active_encounters_summary_groups_by_type(authenticated_client):
    """Ensure the summary endpoint correctly groups active encounters."""
    # 1. Create diverse active encounters
    enc1 = Encounter(patient_id="TEST_P1", encounter_type="SURGICAL", stage="PRE_OP", status="ACTIVE")
    enc2 = Encounter(patient_id="TEST_P2", encounter_type="SURGICAL", stage="INTRA_OP", status="ACTIVE")
    enc3 = Encounter(patient_id="TEST_P3", encounter_type="TELEHEALTH", stage="IN_CONSULTATION", status="ACTIVE")
    enc4 = Encounter(patient_id="TEST_P4", encounter_type="OPD", stage="WAITING_DOCTOR", status="ACTIVE")

    db.session.add_all([enc1, enc2, enc3, enc4])
    db.session.commit()

    # 2. Call the endpoint
    response = authenticated_client.get("/records/api/active_encounters_summary")

    # 3. Assert it returns 200 and valid JSON
    assert response.status_code == 200, f"Expected 200, got {response.status_code}. Response: {response.data}"

    data = response.get_json()
    assert data is not None, "Response was not valid JSON"

    assert data["total_active"] >= 4
    assert data["by_type"]["SURGICAL"]["PRE_OP"] >= 1
    assert data["by_type"]["SURGICAL"]["INTRA_OP"] >= 1
    assert data["by_type"]["TELEHEALTH"]["IN_CONSULTATION"] >= 1
