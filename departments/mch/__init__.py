from flask import Blueprint

from . import routes  # noqa: F401

bp = Blueprint("mch", __name__, url_prefix="/mch")
