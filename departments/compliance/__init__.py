from flask import Blueprint

compliance_bp = Blueprint("compliance", __name__, url_prefix="/compliance")

from . import routes  # noqa: E402, F401
