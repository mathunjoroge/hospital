import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = True  # Enforce CSRF checking
    with app.test_client() as client:
        with app.app_context():
            yield client


def test_post_without_csrf_token_rejected(client):
    """Verify that POST request without a CSRF token returns 400 Bad Request."""
    response = client.post("/login", data={"username": "admin", "password": "password"})
    assert response.status_code == 400
    assert (
        b"CSRF token" in response.data
        or b"The CSRF token is missing" in response.data
        or b"400 Bad Request" in response.data
    )
