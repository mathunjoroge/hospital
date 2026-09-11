from flask import Blueprint

bp = Blueprint("medicine", __name__, template_folder="templates")

from . import (  # noqa: F401
    chat_bot,
    consultations,
    inpatients,
    oncology,
    orders,
    prescriptions,
)
