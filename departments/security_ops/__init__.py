from flask import Blueprint

bp = Blueprint("security_ops", __name__, url_prefix="/security-ops")

from . import routes  # noqa: E402, F401
