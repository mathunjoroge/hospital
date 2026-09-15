from flask import Blueprint

bp = Blueprint("sso", __name__, url_prefix="/auth/sso")

from . import routes  # noqa: E402, F401
