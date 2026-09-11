from flask import Blueprint

bp = Blueprint("clinical_safety", __name__, url_prefix="/clinical-safety")

from . import routes  # noqa: F401
