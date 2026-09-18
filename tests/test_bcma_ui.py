"""
tests/test_bcma_ui.py
──────────────────────────
Unit tests for the Bedside BCMA Scanner Console UI route.
Validates:
  - GET /nursing/bcma/scanner returns 200 with correct template content
  - HTML includes 5-Rights labels
  - Scanner inputs present in the page
  - Override modal markup present
"""

from extensions import db  # noqa: F401 — ensures tables exist


def test_bcma_scanner_ui_route(client, app, admin_user):
    """GET /nursing/bcma/scanner renders BCMA scanner console."""
    resp = client.get("/nursing/bcma/scanner")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    html = resp.data.decode()

    # Core title present
    assert "Bedside BCMA" in html or "BCMA" in html

    # 5 Rights present
    assert "Right Patient" in html
    assert "Right Drug" in html
    assert "Right Dose" in html
    assert "Right Route" in html
    assert "Right Time" in html

    # Barcode inputs
    assert "patient-barcode" in html
    assert "drug-barcode" in html

    # Override modal
    assert "override-modal" in html
    assert "Clinical Override Required" in html or "override" in html.lower()


def test_bcma_scanner_requires_auth(client):
    """GET /nursing/bcma/scanner without auth redirects or returns 401/302."""
    resp = client.get("/nursing/bcma/scanner", follow_redirects=False)
    assert resp.status_code in (
        302,
        401,
        403,
    ), f"Expected redirect/auth error, got {resp.status_code}"


def test_bcma_verify_endpoint_returns_400_missing_fields(client, app, admin_user):
    """POST /nursing/bcma/verify with missing fields returns 400."""
    resp = client.post("/nursing/bcma/verify", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data
