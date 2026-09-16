"""
tests/test_sso_ldap_live.py
─────────────────────────────
Exercises SSOEngine.authenticate_ldap()'s real ldap3 code path — the
Server()/Connection() construction, the sAMAccountName search filter, entry
attribute extraction, and the second bind-as-the-user step that actually
proves the password — against an in-memory LDAP directory (ldap3's MOCK_SYNC
strategy), rather than mocking authenticate_ldap() itself.

Context: ldap3 was not in requirements.txt, so this path had never been
exercised end-to-end; test_sso.py's regression tests only prove that
authenticate_ldap() raises SSOError when ldap3 is unavailable (a real and
useful check, but it can't fail if the working path has a bug — it never
reaches the working path). These tests do.

The only thing mocked is the network transport (MOCK_SYNC replaces LDAP's TCP
layer with an in-memory directory); the search filter, entry parsing, and
bind-as-user logic in sso_engine.py all run for real against it.
"""
import pytest
from ldap3 import MOCK_SYNC, Connection, Server
from ldap3.core.exceptions import LDAPBindError

from departments.sso.sso_engine import SSOEngine, SSOError

SVC_DN = "cn=svc-hims,ou=ServiceAccounts,dc=hospital,dc=local"
SVC_PASSWORD = "svc-secret-pw"
USER_DN = "cn=jdoe,ou=Users,dc=hospital,dc=local"
USER_PASSWORD = "CorrectHorseBatteryStaple!"


@pytest.fixture
def mock_directory(monkeypatch):
    """
    An in-memory LDAP directory seeded with a service account and one real
    user entry, wired in as a drop-in replacement for ldap3.Server /
    ldap3.Connection so sso_engine's own code runs unmodified against it.
    """
    server = Server("mock-ad.hospital.local", get_info=None)

    seed = Connection(server, client_strategy=MOCK_SYNC)
    seed.strategy.add_entry(SVC_DN, {"userPassword": SVC_PASSWORD})
    seed.strategy.add_entry(
        USER_DN,
        {
            "sAMAccountName": "jdoe",
            "userPassword": USER_PASSWORD,
            "mail": "jdoe@hospital.org",
            "displayName": "Jane Doe",
            "memberOf": ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=local"],
        },
    )

    def fake_connection(srv, **kwargs):
        # MOCK_SYNC doesn't establish its bind on construction the way the
        # real network strategy does under auto_bind=True; open()+bind()
        # explicitly and raise the same exception a real failed auto_bind
        # would, so sso_engine's except-LDAPException handling is exercised
        # identically to how it behaves against a real directory.
        auto_bind_requested = kwargs.pop("auto_bind", False)
        kwargs.pop("receive_timeout", None)
        kwargs["client_strategy"] = MOCK_SYNC
        conn = Connection(srv, **kwargs)
        if auto_bind_requested:
            conn.open()
            if not conn.bind():
                raise LDAPBindError(f"mock auto_bind failed: {conn.result}")
        return conn

    monkeypatch.setattr("ldap3.Server", lambda *a, **kw: server)
    monkeypatch.setattr("ldap3.Connection", fake_connection)
    return server


@pytest.fixture
def configured_engine():
    engine = SSOEngine()
    engine.ldap_bind_dn = SVC_DN
    engine.ldap_bind_password = SVC_PASSWORD
    engine.ldap_search_base = "dc=hospital,dc=local"
    return engine


def test_correct_password_authenticates(mock_directory, configured_engine):
    result = configured_engine.authenticate_ldap("jdoe", USER_PASSWORD)
    assert result["username"] == "jdoe"
    assert result["email"] == "jdoe@hospital.org"
    assert result["display_name"] == "Jane Doe"
    assert "CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=local" in result["groups"]


def test_wrong_password_is_rejected(mock_directory, configured_engine):
    """
    The regression this whole fix exists for: the old code accepted every
    password except one hardcoded string. Here, a real bind-as-the-user
    against the directory is what must fail — not a string comparison.
    """
    with pytest.raises(SSOError, match="Invalid Active Directory credentials"):
        configured_engine.authenticate_ldap("jdoe", "totally-wrong-password")


def test_unknown_username_is_rejected(mock_directory, configured_engine):
    with pytest.raises(SSOError, match="Invalid Active Directory credentials"):
        configured_engine.authenticate_ldap("nobody-by-this-name", "whatever")


def test_empty_password_is_rejected_before_any_bind(mock_directory, configured_engine):
    with pytest.raises(SSOError, match="required for LDAP bind"):
        configured_engine.authenticate_ldap("jdoe", "")


def test_service_account_password_does_not_authenticate_as_the_user(
    mock_directory, configured_engine
):
    """The service account's own password must not double as anyone's password."""
    with pytest.raises(SSOError, match="Invalid Active Directory credentials"):
        configured_engine.authenticate_ldap("jdoe", SVC_PASSWORD)


def test_misconfigured_service_account_fails_closed(mock_directory, configured_engine):
    """A wrong service-account password must not silently skip the directory lookup."""
    configured_engine.ldap_bind_password = "wrong-service-account-password"
    with pytest.raises(SSOError, match="Could not reach the directory server"):
        configured_engine.authenticate_ldap("jdoe", USER_PASSWORD)


def test_groups_returned_reflect_the_directory_not_the_username(
    mock_directory, configured_engine
):
    """
    Regression: the old code derived groups by pattern-matching the username
    string ("nurse" in username.lower() -> nursing groups). Confirm the
    fixed path returns the directory's actual memberOf values regardless of
    what the username looks like.
    """
    result = configured_engine.authenticate_ldap("jdoe", USER_PASSWORD)
    assert result["groups"] == ["CN=HIMS-Doctors,OU=Groups,DC=hospital,DC=local"]
