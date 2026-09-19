

def test_csrf_error_handler_returns_custom_json(client, app, admin_user):
    """CSRF errors on API requests return structured JSON with human-friendly message."""
    app.config["WTF_CSRF_ENABLED"] = True
    try:
        response = client.post(
            "/admin/switch_user",
            data={"role": "nursing"},
            headers={"Accept": "application/json"},
        )
        assert response.status_code == 400
        json_data = response.get_json()
        assert json_data["error"] == "Security Verification Failed"
        assert "CSRF" in json_data["message"] or "security" in json_data["message"].lower()
    finally:
        app.config["WTF_CSRF_ENABLED"] = False


def test_unauthorized_access_returns_custom_json(client):
    """Unauthenticated API access returns structured JSON 401 error."""
    response = client.get(
        "/admin/",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 401
    json_data = response.get_json()
    assert json_data["error"] == "Authentication Required"
    assert "signed in" in json_data["message"] or "session" in json_data["message"]


def test_404_not_found_returns_custom_json_for_api(client):
    """404 errors on API requests return custom JSON."""
    response = client.get(
        "/api/non_existent_endpoint_12345",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 404
    json_data = response.get_json()
    assert json_data["error"] == "Resource Not Found"
    assert "could not be found" in json_data["message"]


def test_404_not_found_returns_custom_html_page(client):
    """404 errors on browser requests render custom HTML error template."""
    response = client.get("/non_existent_page_12345")
    assert response.status_code == 404
    assert b"Resource Not Found" in response.data or b"404" in response.data
