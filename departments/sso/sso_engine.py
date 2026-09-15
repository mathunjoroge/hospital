"""
departments/sso/sso_engine.py
──────────────────────────────
Enterprise LDAP / Active Directory & OIDC / SAML SSO Authentication Engine.

Supports:
  1. OpenID Connect (OIDC) / OAuth2 Single Sign-On (Azure AD, Keycloak, Shibboleth, Okta)
  2. Active Directory / LDAP Direct Bind Authentication & Group-to-Role Mapping
  3. Automatic User Provisioning & Group Synchronization
"""
import logging
import os
import re

from departments.models.user import User
from extensions import db

logger = logging.getLogger(__name__)

# Default Active Directory Group to HIMS Role mappings
DEFAULT_AD_GROUP_MAPPINGS = {
    r"CN=HIMS-Admins.*": "admin",
    r"CN=HIMS-Doctors.*": "doctor",
    r"CN=HIMS-Nurses.*": "nursing",
    r"CN=HIMS-Pharmacists.*": "pharmacy",
    r"CN=HIMS-LabTechs.*": "laboratory",
    r"CN=HIMS-Radiologists.*": "imaging",
    r"CN=HIMS-Records.*": "records",
    r"CN=HIMS-Finance.*": "billing",
}


class SSOError(Exception):
    """Base exception for SSO operations."""


class SSOEngine:
    """
    Enterprise SSO engine for OIDC and LDAP/Active Directory integration.
    """

    def __init__(self):
        self.enabled = os.environ.get(
            "ENABLE_SSO", os.environ.get("SSO_ENABLED", "false")
        ).lower() in ("true", "1", "yes")
        self.provider = os.environ.get("SSO_PROVIDER", "oidc").lower()  # oidc | ldap | saml

        # Outbound timeout for IdP/directory calls — an unbounded auth call
        # hangs a worker thread and is a trivial DoS surface.
        self.http_timeout = float(os.environ.get("SSO_HTTP_TIMEOUT", "10"))

        # Auto-provisioning creates a local account for any identity the IdP
        # vouches for. Off by default: operators should pre-create accounts
        # unless they intend every directory user to have a HIMS account.
        self.auto_provision = os.environ.get(
            "SSO_AUTO_PROVISION", "false"
        ).lower() in ("true", "1", "yes")

        # OIDC Configuration — no credential defaults; unset means unconfigured.
        self.oidc_client_id = os.environ.get("OIDC_CLIENT_ID", "hims-enterprise-client")
        self.oidc_client_secret = os.environ.get("OIDC_CLIENT_SECRET", "")
        self.oidc_issuer = os.environ.get("OIDC_ISSUER", "https://login.microsoftonline.com/common/v2.0")
        self.oidc_authorize_url = os.environ.get(
            "OIDC_AUTHORIZE_URL",
            "https://login.microsoftonline.com/common/oauth2/v2.0/authorize"
        )
        self.oidc_token_url = os.environ.get(
            "OIDC_TOKEN_URL",
            "https://login.microsoftonline.com/common/oauth2/v2.0/token"
        )
        self.oidc_userinfo_url = os.environ.get(
            "OIDC_USERINFO_URL",
            "https://graph.microsoft.com/oidc/userinfo"
        )

        # LDAP / Active Directory Configuration
        self.ldap_server = os.environ.get("LDAP_SERVER_URI", "ldap://ad.hospital.org:389")
        self.ldap_bind_dn = os.environ.get("LDAP_BIND_DN", "cn=read-only-admin,dc=hospital,dc=org")
        self.ldap_bind_password = os.environ.get("LDAP_BIND_PASSWORD", "")
        self.ldap_search_base = os.environ.get("LDAP_USER_SEARCH_BASE", "ou=users,dc=hospital,dc=org")

    def get_oidc_authorization_url(self, redirect_uri: str, state: str = "random_state") -> str:
        """
        Build OIDC Authorization URL for Azure AD / Keycloak / Shibboleth.
        """
        from urllib.parse import urlencode

        params = {
            "client_id": self.oidc_client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": "openid profile email groups",
            "state": state,
        }
        return f"{self.oidc_authorize_url}?{urlencode(params)}"

    def process_oidc_callback(self, code: str, redirect_uri: str) -> dict:
        """
        Exchange an OIDC authorization code for tokens at the IdP, then fetch
        the userinfo claims with the resulting access token.

        Fails closed: if the client secret has not been configured, or the IdP
        rejects the code, or the claims carry no username, no identity is
        returned. This function must never synthesise an identity — anything it
        returns is handed straight to login_user().
        """
        if not code:
            raise SSOError("Authorization code is required.")
        if not self.oidc_client_secret:
            raise SSOError("OIDC client secret is not configured.")

        import requests

        try:
            token_resp = requests.post(
                self.oidc_token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.oidc_client_id,
                    "client_secret": self.oidc_client_secret,
                },
                headers={"Accept": "application/json"},
                timeout=self.http_timeout,
            )
        except requests.RequestException as exc:
            raise SSOError(f"Could not reach the identity provider: {exc}") from exc

        if token_resp.status_code != 200:
            logger.warning(
                "OIDC token exchange rejected by IdP (HTTP %s)", token_resp.status_code
            )
            raise SSOError("Identity provider rejected the authorization code.")

        access_token = (token_resp.json() or {}).get("access_token")
        if not access_token:
            raise SSOError("Identity provider returned no access token.")

        try:
            userinfo_resp = requests.get(
                self.oidc_userinfo_url,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
                timeout=self.http_timeout,
            )
        except requests.RequestException as exc:
            raise SSOError(f"Could not reach the userinfo endpoint: {exc}") from exc

        if userinfo_resp.status_code != 200:
            raise SSOError("Identity provider rejected the userinfo request.")

        claims = userinfo_resp.json() or {}
        if not (claims.get("preferred_username") or claims.get("username")):
            raise SSOError("Identity provider returned no username claim.")

        return claims

    def authenticate_ldap(self, username: str, password: str) -> dict:
        """
        Authenticate user credentials directly against Active Directory / LDAP.
        """
        if not username or not password:
            raise SSOError("Username and password are required for LDAP bind.")

        try:
            import ldap3
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise SSOError(
                "LDAP authentication is unavailable: the ldap3 package is not installed."
            ) from exc

        if not self.ldap_bind_password:
            raise SSOError("LDAP service-account credentials are not configured.")

        # 1. Bind as the read-only service account and locate the user entry.
        try:
            server = ldap3.Server(self.ldap_server, get_info=ldap3.NONE)
            svc_conn = ldap3.Connection(
                server,
                user=self.ldap_bind_dn,
                password=self.ldap_bind_password,
                auto_bind=True,
                receive_timeout=self.http_timeout,
            )
        except ldap3.core.exceptions.LDAPException as exc:
            raise SSOError(f"Could not reach the directory server: {exc}") from exc

        try:
            svc_conn.search(
                search_base=self.ldap_search_base,
                search_filter=f"(sAMAccountName={ldap3.utils.conv.escape_filter_chars(username)})",
                attributes=["memberOf", "mail", "displayName"],
            )
            if not svc_conn.entries:
                raise SSOError("Invalid Active Directory credentials.")

            entry = svc_conn.entries[0]
            user_dn = entry.entry_dn
            ad_groups = [str(g) for g in (entry.memberOf.values if "memberOf" in entry else [])]
            email = str(entry.mail) if "mail" in entry and entry.mail else f"{username}@hospital.org"
            display_name = (
                str(entry.displayName)
                if "displayName" in entry and entry.displayName
                else username
            )
        finally:
            svc_conn.unbind()

        # 2. Re-bind as the user with the supplied password. This is the only
        #    thing that actually proves the password — it must not be skipped.
        try:
            user_conn = ldap3.Connection(
                ldap3.Server(self.ldap_server, get_info=ldap3.NONE),
                user=user_dn,
                password=password,
                receive_timeout=self.http_timeout,
            )
            if not user_conn.bind():
                raise SSOError("Invalid Active Directory credentials.")
            user_conn.unbind()
        except ldap3.core.exceptions.LDAPException as exc:
            raise SSOError("Invalid Active Directory credentials.") from exc

        return {
            "username": username,
            "email": email,
            "display_name": display_name,
            "groups": ad_groups,
        }

    def map_ad_groups_to_role(self, groups: list[str]) -> str:
        """
        Map Active Directory / OIDC group membership strings to HIMS app role.
        """
        for group in groups:
            for pattern, role in DEFAULT_AD_GROUP_MAPPINGS.items():
                if re.match(pattern, group, re.IGNORECASE):
                    return role
        # No mapped group => no role. Falling back to a privileged default
        # ("doctor") granted read/write/prescribe to any directory identity.
        raise SSOError(
            "No HIMS role is mapped to this account's directory groups."
        )

    def provision_or_sync_user(self, user_info: dict) -> User:
        """
        Provision a new user or update an existing user's roles and attributes from SSO payload.
        """
        username = user_info.get("preferred_username") or user_info.get("username")
        groups = user_info.get("groups", [])

        if not username:
            raise SSOError("User payload missing username attribute.")

        role = self.map_ad_groups_to_role(groups)

        user = User.query.filter_by(username=username).first()

        if not user:
            if not self.auto_provision:
                raise SSOError(
                    "No local account exists for this identity and SSO "
                    "auto-provisioning is disabled."
                )
            logger.info(f"Auto-provisioning new SSO user: {username} with role {role}")
            user = User(
                username=username,
                role=role,
            )
            # Set unguessable password for SSO-provisioned accounts
            import secrets

            from werkzeug.security import generate_password_hash
            user.password = generate_password_hash(secrets.token_urlsafe(24))
            db.session.add(user)
        else:
            logger.info(f"Syncing existing SSO user: {username}")
            user.role = role

        db.session.commit()
        return user
