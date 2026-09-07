from flask import Blueprint

bp = Blueprint("system_ops", __name__, url_prefix="/system-ops")

from . import routes  # noqa: E402, F401
