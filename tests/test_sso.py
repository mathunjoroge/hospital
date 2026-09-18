"""
tests/test_sso.py
──────────────────
Enterprise LDAP / Active Directory & OIDC Single Sign-On (SSO).

These tests replace an earlier suite that asserted the *simulated* engine
behaved correctly — it did, and that simulation was an authentication bypass:
authenticate_ldap() accepted every password except the literal string
"WrongPassword!", and process_oidc_callback() ignored the authorization code
and returned a hardcoded identity. Both then called login_user().

The invariants below are therefore written negatively where it matters: the
blueprint must be unreachable unless explicitly enabled, and no code path may
produce an authenticated session without a real IdP saying so.
"""

from unittest.mock import MagicMock, patch

import pytest

from departments.sso.sso_engine import SSOEngine, SSOError


@pytest.fixture
def sso_on(app):
    """Enable the SSO blueprint for tests that need to reach the endpoints."""
    prev = app.config.get("ENABLE_SSO", False)
    app.config["ENABLE_SSO"] = True
    yield app
    app.config["ENABLE_SSO"] = prev


# ── Feature flag quarantine ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/auth/sso/status"),
        ("get", "/auth/sso/login"),
        ("get", "/auth/sso/callback?code=abc"),
        ("post", "/auth/sso/ldap"),
    ],
)
def test_sso_endpoints_disabled_by_default(client, app, method, path):
    """Every SSO endpoint is 403 FEATURE_DISABLED unless ENABLE_SSO is set."""
    app.config["ENABLE_SSO"] = False
    response = getattr(client, method)(path)
    assert response.status_code == 403
    assert response.get_json()["code"] == "FEATURE_DISABLED"


def test_sso_engine_defaults_to_disabled(monkeypatch):
    """SSOEngine must not default to enabled when no env vars are present."""
    monkeypatch.delenv("ENABLE_SSO", raising=False)
    monkeypatch.delenv("SSO_ENABLED", raising=False)
    assert SSOEngine().enabled is False


def test_no_default_credentials():
    """Unconfigured secrets must be empty, never a shipped placeholder value."""
    engine = SSOEngine()
    assert engine.oidc_client_secret == ""
    assert engine.ldap_bind_password == ""


# ── The bypass itself, asserted as a regression guard ─────────────────────


def test_ldap_arbitrary_password_is_rejected(client, sso_on):
    """
    Regression: POST /auth/sso/ldap with an arbitrary password previously
    returned 200, auto-created the account, mapped 'admin' in the username to
    the admin role, and logged the caller in.
    """
    response = client.post(
        "/auth/sso/ldap",
        json={"username": "attacker_admin", "password": "literally-anything"},
    )
    assert response.status_code == 401

    from departments.models.user import User

    assert User.query.filter_by(username="attacker_admin").first() is None


def test_forged_ldap_login_does_not_grant_admin(client, sso_on):
    """A rejected LDAP bind must leave the session anonymous."""
    client.post("/auth/sso/ldap", json={"username": "evil_admin", "password": "x"})
    response = client.get("/admin/", follow_redirects=False)
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_oidc_callback_with_forged_code_is_rejected(client, sso_on):
    """A fabricated authorization code must not yield an identity."""
    response = client.get(
        "/auth/sso/callback?code=totally-made-up",
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 401
    assert (
        response.get_json()["status"] != "success"
        if response.get_json().get("status")
        else True
    )


def test_oidc_requires_configured_client_secret():
    engine = SSOEngine()
    with pytest.raises(SSOError, match="client secret is not configured"):
        engine.process_oidc_callback(code="abc", redirect_uri="https://h/cb")


def test_oidc_rejects_empty_code():
    with pytest.raises(SSOError, match="Authorization code is required"):
        SSOEngine().process_oidc_callback(code="", redirect_uri="https://h/cb")


# ── Happy path, with the IdP transport mocked ─────────────────────────────


def _idp_responses(claims):
    token = MagicMock(status_code=200)
    token.json.return_value = {"access_token": "at_123"}
    userinfo = MagicMock(status_code=200)
    userinfo.json.return_value = claims
    return token, userinfo


def test_oidc_happy_path_returns_idp_claims(monkeypatch):
    """With a real 200 from the IdP, the engine returns the IdP's claims."""
    engine = SSOEngine()
    engine.oidc_client_secret = "configured-secret"
    claims = {
        "sub": "azure-ad-user-12345",
        "preferred_username": "dr.johnson",
        "groups": ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"],
    }
    token, userinfo = _idp_responses(claims)
    with patch("requests.post", return_value=token), patch(
        "requests.get", return_value=userinfo
    ):
        assert engine.process_oidc_callback("real_code", "https://h/cb") == claims


def test_oidc_rejects_idp_error_response():
    engine = SSOEngine()
    engine.oidc_client_secret = "configured-secret"
    with patch("requests.post", return_value=MagicMock(status_code=400)):
        with pytest.raises(SSOError, match="rejected the authorization code"):
            engine.process_oidc_callback("bad_code", "https://h/cb")


def test_oidc_rejects_claims_without_username():
    engine = SSOEngine()
    engine.oidc_client_secret = "configured-secret"
    token, userinfo = _idp_responses({"sub": "no-username-here"})
    with patch("requests.post", return_value=token), patch(
        "requests.get", return_value=userinfo
    ):
        with pytest.raises(SSOError, match="no username claim"):
            engine.process_oidc_callback("real_code", "https://h/cb")


def test_oidc_calls_are_timeout_bounded():
    """An unbounded auth call hangs a worker; both legs must pass a timeout."""
    engine = SSOEngine()
    engine.oidc_client_secret = "configured-secret"
    token, userinfo = _idp_responses({"preferred_username": "dr.johnson", "groups": []})
    with patch("requests.post", return_value=token) as post, patch(
        "requests.get", return_value=userinfo
    ) as get:
        try:
            engine.process_oidc_callback("real_code", "https://h/cb")
        except SSOError:
            pass
        assert post.call_args.kwargs["timeout"] == engine.http_timeout
        assert get.call_args.kwargs["timeout"] == engine.http_timeout


# ── Role mapping & provisioning ───────────────────────────────────────────


def test_ad_group_mapping_logic():
    engine = SSOEngine()
    assert (
        engine.map_ad_groups_to_role(["CN=HIMS-Admins,OU=Groups,DC=hospital,DC=org"])
        == "admin"
    )
    assert (
        engine.map_ad_groups_to_role(["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"])
        == "doctor"
    )
    assert (
        engine.map_ad_groups_to_role(["CN=HIMS-Nurses,OU=Groups,DC=hospital,DC=org"])
        == "nursing"
    )
    assert (
        engine.map_ad_groups_to_role(
            ["CN=HIMS-Pharmacists,OU=Groups,DC=hospital,DC=org"]
        )
        == "pharmacy"
    )


def test_unmapped_groups_do_not_fall_back_to_a_privileged_role():
    """Regression: unmapped groups previously returned 'doctor'."""
    with pytest.raises(SSOError, match="No HIMS role is mapped"):
        SSOEngine().map_ad_groups_to_role(
            ["CN=Building-Cleaners,OU=Groups,DC=hospital,DC=org"]
        )


def test_no_groups_at_all_is_rejected():
    with pytest.raises(SSOError, match="No HIMS role is mapped"):
        SSOEngine().map_ad_groups_to_role([])


def test_auto_provisioning_is_off_by_default(app):
    engine = SSOEngine()
    assert engine.auto_provision is False
    with pytest.raises(SSOError, match="auto-provisioning is disabled"):
        engine.provision_or_sync_user(
            {
                "preferred_username": "brand.new.user",
                "groups": ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"],
            }
        )


def test_auto_provisioning_when_explicitly_enabled(app):
    engine = SSOEngine()
    engine.auto_provision = True
    user = engine.provision_or_sync_user(
        {
            "preferred_username": "new.doctor",
            "groups": ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"],
        }
    )
    assert user.username == "new.doctor"
    assert user.role == "doctor"


def test_provisioned_account_has_no_usable_password(app):
    """SSO accounts must not be loggable-into via the local password form."""
    engine = SSOEngine()
    engine.auto_provision = True
    user = engine.provision_or_sync_user(
        {
            "preferred_username": "sso.only",
            "groups": ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=org"],
        }
    )
    assert user.password
    from werkzeug.security import check_password_hash

    assert not check_password_hash(user.password, "")


def test_status_endpoint_when_enabled(client, sso_on):
    response = client.get("/auth/sso/status")
    assert response.status_code == 200
    data = response.get_json()
    assert {"sso_enabled", "provider", "supported_methods"} <= set(data)


def test_status_endpoint_leaks_no_secrets(client, sso_on):
    body = client.get("/auth/sso/status").get_data(as_text=True).lower()
    assert "secret" not in body
    assert "password" not in body


def test_oidc_login_redirect(client, sso_on):
    response = client.get("/auth/sso/login", follow_redirects=False)
    assert response.status_code == 302
    assert "login.microsoftonline.com" in response.headers["Location"]
