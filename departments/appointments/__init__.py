from flask import Blueprint

bp = Blueprint("appointments", __name__, url_prefix="/appointments")

from . import routes  # noqa: F401
