"""
tests/test_fhir_terminology_server.py
──────────────────────────────────────
Test suite for FHIR R4 Terminology Server and Autocomplete Search API.
"""

from departments.medicine.terminology_server import FHIRTerminologyServer
from departments.models.terminology import ICD10Code, LoincCode, SnomedCode
from extensions import db


def test_fhir_terminology_server_direct_lookup(app):
    """Test FHIRTerminologyServer.lookup_code directly."""
    with app.app_context():
        # Insert test records
        icd = ICD10Code(code="J00", description="Acute nasopharyngitis [common cold]", chapter="Respiratory")
        snomed = SnomedCode(code="404684003", description="Clinical finding")
        loinc = LoincCode(code="8302-2", description="Body height")
        db.session.add_all([icd, snomed, loinc])
        db.session.commit()

        # Test ICD-10 lookup
        res_icd = FHIRTerminologyServer.lookup_code("http://hl7.org/fhir/sid/icd-10", "J00")
        assert res_icd["resourceType"] == "Parameters"
        params = {p["name"]: p.get("valueString") or p.get("valueCode") or p.get("valueUri") for p in res_icd["parameter"]}
        assert params["code"] == "J00"
        assert "common cold" in params["display"]

        # Test SNOMED lookup
        res_snomed = FHIRTerminologyServer.lookup_code("SNOMED", "404684003")
        assert res_snomed["resourceType"] == "Parameters"
        params_sn = {p["name"]: p.get("valueString") or p.get("valueCode") for p in res_snomed["parameter"]}
        assert params_sn["code"] == "404684003"
        assert params_sn["display"] == "Clinical finding"

        # Test LOINC lookup
        res_loinc = FHIRTerminologyServer.lookup_code("loinc", "8302-2")
        assert res_loinc["resourceType"] == "Parameters"

        # Test not found
        res_nf = FHIRTerminologyServer.lookup_code("ICD10", "UNKNOWN999")
        assert res_nf.get("error") is True
        assert res_nf.get("status") == 404


def test_fhir_terminology_server_validate_code(app):
    """Test FHIRTerminologyServer.validate_code directly."""
    with app.app_context():
        icd = ICD10Code(code="I10", description="Essential (primary) hypertension", chapter="Cardiovascular")
        db.session.add(icd)
        db.session.commit()

        # Valid code and display
        v1 = FHIRTerminologyServer.validate_code("http://hl7.org/fhir/sid/icd-10", "I10", "hypertension")
        params1 = {p["name"]: p["valueBoolean"] if "valueBoolean" in p else p.get("valueString") for p in v1["parameter"]}
        assert params1["result"] is True

        # Invalid code
        v2 = FHIRTerminologyServer.validate_code("ICD10", "INVALID_CODE")
        params2 = {p["name"]: p["valueBoolean"] if "valueBoolean" in p else p.get("valueString") for p in v2["parameter"]}
        assert params2["result"] is False

        # Mismatched display
        v3 = FHIRTerminologyServer.validate_code("ICD10", "I10", "completely wrong title")
        params3 = {p["name"]: p["valueBoolean"] if "valueBoolean" in p else p.get("valueString") for p in v3["parameter"]}
        assert params3["result"] is False



def test_fhir_terminology_server_search(app):
    """Test FHIRTerminologyServer.search_terms autocomplete functionality."""
    with app.app_context():
        icd = ICD10Code(code="R50.9", description="Fever, unspecified", chapter="General")
        db.session.add(icd)
        db.session.commit()

        res = FHIRTerminologyServer.search_terms("fever", system="icd10")
        assert len(res) >= 1
        assert res[0]["code"] == "R50.9"
        assert "Fever" in res[0]["description"]


def test_fhir_lookup_endpoint(client, app):
    """Test GET and POST /api/fhir/R4/CodeSystem/$lookup API endpoint."""
    with app.app_context():
        icd = ICD10Code(code="J00", description="Acute nasopharyngitis [common cold]", chapter="Respiratory")
        db.session.add(icd)
        db.session.commit()

    # GET request
    resp = client.get("/api/fhir/R4/CodeSystem/$lookup?system=icd10&code=J00")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["resourceType"] == "Parameters"

    # POST request with JSON payload
    resp_post = client.post(
        "/api/fhir/R4/CodeSystem/$lookup",
        json={"system": "http://hl7.org/fhir/sid/icd-10", "code": "J00"},
    )
    assert resp_post.status_code == 200

    # Missing parameters -> 400
    resp_bad = client.get("/api/fhir/R4/CodeSystem/$lookup")
    assert resp_bad.status_code == 400


def test_fhir_validate_code_endpoint(client, app):
    """Test GET and POST /api/fhir/R4/CodeSystem/$validate-code API endpoint."""
    with app.app_context():
        icd = ICD10Code(code="I10", description="Essential (primary) hypertension", chapter="Cardiovascular")
        db.session.add(icd)
        db.session.commit()

    resp = client.get("/api/fhir/R4/CodeSystem/$validate-code?system=icd10&code=I10&display=hypertension")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["resourceType"] == "Parameters"
    res_val = [p for p in data["parameter"] if p["name"] == "result"][0]
    assert res_val["valueBoolean"] is True


def test_api_terminology_search_endpoint(client, app):
    """Test GET /api/terminology/search API endpoint."""
    with app.app_context():
        icd = ICD10Code(code="E11.9", description="Type 2 diabetes mellitus without complications", chapter="Endocrine")
        db.session.add(icd)
        db.session.commit()

    resp = client.get("/api/terminology/search?q=diabetes&system=ICD10")
    assert resp.status_code == 200
    results = resp.get_json()
    assert isinstance(results, list)
    assert len(results) >= 1
    assert any("diabetes" in item["description"].lower() for item in results)
