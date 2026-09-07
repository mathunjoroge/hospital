from flask import Blueprint

from . import routes  # noqa: F401

bp = Blueprint("appointments", __name__, url_prefix="/appointments")
