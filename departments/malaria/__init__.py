from flask import Blueprint

bp = Blueprint("malaria", __name__, url_prefix="/malaria")

from . import routes  # noqa: F401
