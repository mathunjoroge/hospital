from flask import Blueprint

bp = Blueprint("public_health", __name__, url_prefix="/public-health")

from . import routes  # noqa: E402, F401
