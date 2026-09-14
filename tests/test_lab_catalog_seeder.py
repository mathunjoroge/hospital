"""
tests/test_lab_catalog_seeder.py
──────────────────────────────────
Unit tests for lab catalog seeder and LOINC mapping.
"""

from departments.laboratory.lab_catalog_seeder import seed_lab_test_catalog
from departments.models.laboratory import LabResultTemplate
from departments.models.medicine import LabTest


def test_seed_lab_test_catalog(app):
    """Verify that seed_lab_test_catalog populates standard lab tests and templates."""
    with app.app_context():
        count = seed_lab_test_catalog()
        assert count > 0

        # Check specific test
        fbc = LabTest.query.filter_by(test_name="Full Blood Count (FBC/CBC)").first()
        assert fbc is not None
        assert fbc.loinc_code == "58410-2"
        assert fbc.cost == 1500.0

        # Check result templates
        templates = LabResultTemplate.query.filter_by(test_id=fbc.id).all()
        assert len(templates) == 5
        param_names = [t.parameter_name for t in templates]
        assert "Hemoglobin" in param_names
        assert "Platelets" in param_names


def test_seed_lab_test_catalog_idempotent(app):
    """Verify that running seed_lab_test_catalog twice is idempotent."""
    with app.app_context():
        seed_lab_test_catalog()
        second_count = seed_lab_test_catalog()
        assert second_count == 0
