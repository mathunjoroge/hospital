from datetime import datetime, timezone

import pytest

from app import app
from departments.hiv_art.models import (
    ARTRegimen,
)
from extensions import db


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.app_context():
        db.create_all()
        with app.test_client() as client:
            yield client
        db.session.remove()
        db.drop_all()


@pytest.fixture
def sample_regimen():
    """Create a sample ART regimen for testing."""
    regimen = ARTRegimen(
        regimen_code="TDF/3TC/EFV",
        regimen_name="Tenofovir/Lamivudine/Efavirenz",
        line_of_therapy=1,
        arv_drugs="Tenofovir, Lamivudine, Efavirenz",
        is_preferred=True,
        effective_from=datetime.now(timezone.utc).date()
    )
    db.session.add(regimen)
    db.session.commit()
    return regimen


def test_art_regimen_creation(client, sample_regimen):
    """Test that ART regimens can be created and retrieved."""
    assert sample_regimen.id is not None
    assert sample_regimen.regimen_code == "TDF/3TC/EFV"
    assert sample_regimen.line_of_therapy == 1
    assert sample_regimen.is_preferred == True

    # Test retrieval from DB
    retrieved = ARTRegimen.query.get(sample_regimen.id)
    assert retrieved is not None
    assert retrieved.regimen_name == "Tenofovir/Lamivudine/Efavirenz"


def test_art_enrollment_creation(client, sample_regimen):
    """Test that ART enrollments can be created."""
    from departments.hiv_art.engine import create_art_enrollment

    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        baseline_who_stage=2,
        art_start_date=datetime.now(timezone.utc),
        facility_enrolled_at="Test Clinic",
        current_regimen_id=sample_regimen.id
    )

    assert success == True
    assert enrollment is not None
    assert enrollment.patient_id == "PAT001"
    assert enrollment.art_number == "ART0001"
    assert enrollment.baseline_cd4 == 350
    assert enrollment.baseline_who_stage == 2
    assert enrollment.current_regimen_id == sample_regimen.id

    # Test that we can't enroll the same patient twice
    success2, message2, _ = create_art_enrollment(
        patient_id="PAT001",  # Same patient
        art_number="ART0002",
        baseline_cd4=400
    )

    assert success2 == False
    assert "already has an active ART enrollment" in message2


def test_art_enrollment_duplicate_art_number(client, sample_regimen):
    """Test that ART numbers must be unique."""
    from departments.hiv_art.engine import create_art_enrollment

    # Create first enrollment
    success1, message1, enrollment1 = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success1 == True

    # Try to create second enrollment with same ART number
    success2, message2, _ = create_art_enrollment(
        patient_id="PAT002",  # Different patient
        art_number="ART0001",  # Same ART number
        baseline_cd4=400,
        current_regimen_id=sample_regimen.id
    )

    assert success2 == False
    assert "already assigned to another patient" in message2


def test_adherence_visit_recording(client, sample_regimen):
    """Test recording adherence visits."""
    from departments.hiv_art.engine import create_art_enrollment, record_adherence_visit

    # Create enrollment first
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success == True

    # Record adherence visit
    success2, message2, visit = record_adherence_visit(
        enrollment_id=enrollment.id,
        pills_dispensed=30,
        pills_returned=5,  # Took 25 out of 30
        visit_date=datetime.now(timezone.utc)
    )

    assert success2 == True
    assert visit is not None
    assert visit.pills_dispensed == 30
    assert visit.pills_returned == 5
    # Adherence should be (25/30)*100 = 83.33%
    assert abs(visit.adherence_percentage - 83.33) < 0.1
    assert visit.adherence_category == "fair"  # 80-94% is fair


def test_viral_load_recording(client, sample_regimen):
    """Test recording viral load results."""
    from departments.hiv_art.engine import create_art_enrollment, record_viral_load

    # Create enrollment first
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success == True

    # Record detectable viral load
    success2, message2, vl = record_viral_load(
        enrollment_id=enrollment.id,
        viral_load_copies=12500,
        test_type="routine"
    )

    assert success2 == True
    assert vl is not None
    assert vl.viral_load_copies == 12500
    assert vl.test_type == "routine"

    # Record undetectable viral load
    success3, message3, vl2 = record_viral_load(
        enrollment_id=enrollment.id,
        viral_load_copies=None,  # Undetectable
        test_type="routine"
    )

    assert success3 == True
    assert vl2 is not None
    assert vl2.viral_load_copies is None


def test_cd4_recording(client, sample_regimen):
    """Test recording CD4 count results."""
    from departments.hiv_art.engine import create_art_enrollment, record_cd4_count

    # Create enrollment first
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success == True

    # Record CD4 count
    success2, message2, cd4 = record_cd4_count(
        enrollment_id=enrollment.id,
        cd4_count=420,
        cd4_percent=25.0
    )

    assert success2 == True
    assert cd4 is not None
    assert cd4.cd4_count == 420
    assert cd4.cd4_percent == 25.0


def test_who_stage_recording(client, sample_regimen):
    """Test recording WHO stage assessments."""
    from departments.hiv_art.engine import create_art_enrollment, record_who_stage

    # Create enrollment first
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success == True

    # Record WHO stage
    success2, message2, who = record_who_stage(
        enrollment_id=enrollment.id,
        who_stage=3,
        defining_conditions="Weight loss, chronic diarrhea"
    )

    assert success2 == True
    assert who is not None
    assert who.who_stage == 3
    assert who.defining_conditions == "Weight loss, chronic diarrhea"


def test_regimen_change(client, sample_regimen):
    """Test changing a patient's ART regimen."""
    from departments.hiv_art.engine import create_art_enrollment, update_art_regimen

    # Create a second regimen for testing line change
    regimen2 = ARTRegimen(
        regimen_code="AZT/3TC/LPV/r",
        regimen_name="Zidovudine/Lamivudine/Lopinavir/ritonavir",
        line_of_therapy=2,  # Second line
        arv_drugs="Zidovudine, Lamivudine, Lopinavir/ritonavir",
        is_preferred=False,
        is_alternative=True,
        effective_from=datetime.now(timezone.utc).date()
    )
    db.session.add(regimen2)
    db.session.commit()

    # Create enrollment with first line regimen
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id  # First line regimen
    )

    assert success == True
    assert enrollment.current_regimen.line_of_therapy == 1

    # Change to second line regimen (requires clinical reason)
    success2, message2, updated_enrollment = update_art_regimen(
        enrollment_id=enrollment.id,
        new_regimen_id=regimen2.id,
        change_reason="Treatment failure after 6 months on first line regimen",
        approved_by="DR001",
        encounter_id=1
    )

    assert success2 == True
    assert updated_enrollment is not None
    assert updated_enrollment.current_regimen_id == regimen2.id
    assert updated_enrollment.current_regimen.line_of_therapy == 2


def test_model_relationships(client, sample_regimen):
    """Test that model relationships work correctly."""
    from departments.hiv_art.engine import (
        create_art_enrollment,
        record_adherence_visit,
        record_viral_load,
    )

    # Create enrollment
    success, message, enrollment = create_art_enrollment(
        patient_id="PAT001",
        art_number="ART0001",
        baseline_cd4=350,
        current_regimen_id=sample_regimen.id
    )

    assert success == True

    # Add adherence visit
    success2, message2, visit = record_adherence_visit(
        enrollment_id=enrollment.id,
        pills_dispensed=20,
        pills_returned=2
    )

    assert success2 == True

    # Add viral load
    success3, message3, vl = record_viral_load(
        enrollment_id=enrollment.id,
        viral_load_copies=5000
    )

    assert success3 == True

    # Test relationships
    assert len(enrollment.adherence_visits.all()) == 1
    assert enrollment.adherence_visits.first().id == visit.id

    assert len(enrollment.viral_loads.all()) == 1
    assert enrollment.viral_loads.first().id == vl.id

    # Test reverse relationship
    assert visit.enrollment.id == enrollment.id
    assert vl.enrollment.id == enrollment.id


def test_art_regimen_formulary_functions(client, sample_regimen):
    """Test formulary-related functions."""
    from departments.hiv_art.engine import (
        get_art_formulary,
        is_regimen_valid,
        log_formulary_change,
    )

    # Test getting formulary
    formulary = get_art_formulary()
    assert len(formulary) >= 1
    assert any(r.id == sample_regimen.id for r in formulary)

    # Test regimen validation
    assert is_regimen_valid(sample_regimen.id) == True

    # Test with non-existent regimen
    assert is_regimen_valid("non-existent-id") == False

    # Test logging (should not raise exception)
    result = log_formulary_change(
        regimen_id=sample_regimen.id,
        change_type="test",
        changed_by="TEST_USER",
        change_notes="This is a test log entry"
    )
    assert result == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
