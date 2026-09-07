from flask import Blueprint

from . import routes  # noqa: F401

bp = Blueprint("clinical_safety", __name__, url_prefix="/clinical-safety")
